import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.init import xavier_uniform_initialization
import math
import numpy as np
from multiprocessing import Pool, cpu_count
import json

class CollaborativePrototypeRefiner(nn.Module):
    """Overlap-supervised residual prototype adjustment, used during training only."""

    def __init__(self, embedding_dim, hidden_dim=None):
        super().__init__()
        h = hidden_dim or embedding_dim
        self.fc1 = nn.Linear(embedding_dim, h)
        self.fc2 = nn.Linear(h, embedding_dim)

    def reset_residual(self):
        # Start as an identity mapping: z = x + residual(x).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, prototype):
        return prototype + self.fc2(F.relu(self.fc1(prototype)))


class LSDCDR(GeneralRecommender):
    """LSDCDR: diffusion reconstruction + independently supervised prototype refiner.

    Objective:
      * diff_tgt: reconstruct *target item embeddings* for observed overlap pairs
        and pseudo target pairs of source-only users (no BPR or ranking loss).
      * prototype_refiner: sampled-softmax on real overlap target interactions;
        its inputs are stop-gradient EMA diffusion prototypes, with target item
        embeddings detached in this auxiliary objective.
      * Refiner affects diffusion solely by rebuilding pseudo target pairs.
      * Inference: ONLINE diffusion only; never uses prototype_refiner or EMA.

    Trainer contract: calculate_loss() returns ONE scalar so existing optimizers
    can step the diffusion/encoder and refiner parameters jointly; their graphs
    are independent. post_batch_processing() MUST be invoked AFTER optimizer.step()
    to update EMA and, when due, refresh pseudo interactions.
    """

    def __init__(self, config, dataloader):
        super().__init__(config, dataloader)
        self.config = config
        self.embedding_dim = int(config['embedding_dim'])
        self.diff_weight = float(config.get('diff_weight', 1.0))
        self.pseudo_diff_weight = float(config.get('pseudo_diff_weight', 1.0))
        self.refiner_weight = float(config.get('refiner_weight', 1.0))
        self.refiner_temperature = float(config.get('refiner_temperature', 0.2))
        self.refiner_num_neg = int(config.get('refiner_num_neg', 32))
        self.refiner_train_interval = int(config.get('refiner_train_interval', 1))
        self.pseudo_start_epoch = int(config.get('pseudo_start_epoch', 0))
        self.refiner_start_epoch = int(config.get('refiner_start_epoch', self.pseudo_start_epoch))
        self.pseudo_update_interval = int(config.get('pseudo_update_interval', 0))
        self.pseudo_generation_batch_size = int(config.get('pseudo_generation_batch_size', 512))
        # Direct, causally clean ablations (w/o IR, w/o CR, w/o EMA).
        self.use_iterative_refinement = bool(config.get('use_iterative_refinement', True))
        self.use_collaborative_refiner = bool(config.get('use_collaborative_refiner', True))
        self.use_ema_for_prototypes = bool(config.get('use_ema_for_prototypes', True))
        if self.refiner_temperature <= 0 or self.refiner_num_neg <= 0:
            raise ValueError('refiner_temperature and refiner_num_neg must be > 0')
        if self.refiner_train_interval <= 0:
            raise ValueError('refiner_train_interval must be > 0')

        self.emb_user = nn.Embedding(
            self.num_users_src + self.num_users_tgt - self.num_users_overlap + 1,
            self.embedding_dim, padding_idx=0
        )
        self.emb_item_src = nn.Embedding(self.num_items_src + 1,
                                         self.embedding_dim, padding_idx=0)
        self.emb_item_tgt = nn.Embedding(self.num_items_tgt + 1,
                                         self.embedding_dim, padding_idx=0)

        # Existing, offline LLM + semantic retrieval initialization is unchanged.
        semantic_emb = torch.from_numpy(
            dataloader.get_modality_embs()['CrossDomain_semantics_both_tag']
        ).float()
        num_users = self.num_users_src + self.num_users_tgt - self.num_users_overlap
        assert semantic_emb.shape[0] == num_users + self.num_items_src + self.num_items_tgt, \
            'Semantic embedding size mismatch.'
        user_sem = semantic_emb[:num_users]
        src_item_sem = semantic_emb[num_users:num_users + self.num_items_src]
        tgt_item_sem = semantic_emb[num_users + self.num_items_src:]
        for name, embeddings in (
            ('user_text_emb', user_sem),
            ('src_item_text_emb', src_item_sem),
            ('tgt_item_text_emb', tgt_item_sem),
        ):
            self.register_buffer(name, torch.cat([torch.zeros_like(embeddings[:1]), embeddings], dim=0))

        self.src_only_pseudo_pos = self._build_src_only_pseudo_positives()
        self.pseudo_ui_pairs = [
            (u, item) for u, items in self.src_only_pseudo_pos.items()
            for item in sorted(items)
        ]

        # Only one (target-domain) online diffusion model is needed.
        self.diff_tgt = Diffusion(config)
        self.src_interaction_agg = InteractionAggregator(config)
        self.prototype_refiner = CollaborativePrototypeRefiner(
            self.embedding_dim, config.get('refiner_hidden_dim', self.embedding_dim)
        )

        (self.history_src_user_src,
         self.history_src_user_tgt,
         self.history_tgt_user_src,
         self.history_tgt_user_tgt) = self._build_padded_history(dataloader)
        # Never use target-domain histories as a diffusion condition:
        # test/cold users must have ZERO target observations in training.
        self.overlap_target_positives = {
            int(u): set(map(int, items))
            for u, items in dataloader.dataset.positive_items_tgt.items()
            if 1 <= int(u) <= self.num_users_overlap
        }

        self.apply(xavier_uniform_initialization)
        with torch.no_grad():
            self.emb_user.weight[0].zero_()
            self.emb_item_src.weight[0].zero_()
            self.emb_item_tgt.weight[0].zero_()
        self.prototype_refiner.reset_residual()

        # Optional use of externally pretrained target item embeddings.
        # Set True ONLY if actual pretrained weights are subsequently loaded.
        if config.get('freeze_tgt_item_embeddings', False):
            self.emb_item_tgt.weight.requires_grad_(False)

        # EMA must exist before training: prototype refiner uses it from epoch 0.
        import copy
        self.diff_tgt_ema = copy.deepcopy(self.diff_tgt)
        self.diff_tgt_ema.requires_grad_(False)
        self.diff_tgt_ema.eval()
        self.ema_initialized = True
        self.global_step = 0
        self.pseudo_refinement_count = 0
        self.last_loss_breakdown = {}

    @torch.no_grad()
    def _build_src_only_pseudo_positives(self):
        top_k = min(int(self.config['top_k_pos']), self.num_items_tgt)
        if top_k <= 0:
            raise ValueError('top_k_pos must be positive')
        start, end = self.num_users_overlap + 1, self.num_users_src + 1
        user_emb = self.user_text_emb[start:end]
        item_emb = self.tgt_item_text_emb[1:]
        if not user_emb.numel():
            return {}
        topk_idx = (user_emb @ item_emb.T).topk(k=top_k, dim=1).indices + 1
        return {uid: set(row.tolist()) for uid, row in zip(range(start, end), topk_idx)}

    def _build_padded_history(self, dataloader):
        L = int(self.config['history_len'])
        h_ss = torch.zeros((self.num_users_src + 1, L), dtype=torch.long, device=self.device)
        h_st = torch.zeros((self.num_users_src + 1, L), dtype=torch.long, device=self.device)
        h_ts = torch.zeros((self.num_users_tgt + 1, L), dtype=torch.long, device=self.device)
        h_tt = torch.zeros((self.num_users_tgt + 1, L), dtype=torch.long, device=self.device)
        for u, items in dataloader.dataset.positive_items_src.items():
            ids = list(items)[-L:]
            if ids:
                h_ss[u, :len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                if u <= self.num_users_overlap:
                    h_ts[u, :len(ids)] = h_ss[u, :len(ids)]
        for u, items in dataloader.dataset.positive_items_tgt.items():
            ids = list(items)[-L:]
            if ids:
                h_tt[u, :len(ids)] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                if u <= self.num_users_overlap:
                    h_st[u, :len(ids)] = h_tt[u, :len(ids)]
        return h_ss, h_st, h_ts, h_tt

    @staticmethod
    def batch_random_mask(seq, mask_rate, min_keep=1):
        if mask_rate <= 0:
            return seq
        valid = seq != 0
        cnt = valid.sum(dim=1)
        keep = ((1.0 - mask_rate) * cnt.float()).long().clamp(min=min_keep)
        keep = torch.minimum(keep, cnt)
        rand = torch.rand(seq.shape, device=seq.device).masked_fill(~valid, 2.0)
        order = rand.argsort(dim=1)
        mask_order = (torch.arange(seq.size(1), device=seq.device)[None, :] < keep[:, None])
        keep_mask = torch.zeros_like(valid)
        keep_mask.scatter_(1, order, mask_order)
        return seq.masked_fill(~keep_mask, 0)

    def _src_condition(self, global_users, src_histories, training_mask=False):
        if training_mask:
            src_histories = self.batch_random_mask(
                src_histories, float(self.config.get('mask_rate', 0.0)))
        src_item_emb = self.emb_item_src(src_histories)
        query = self.emb_user(global_users)
        return self.src_interaction_agg(src_item_emb, query)

    def _target_reconstruction(self, users, pos_items, src_histories):
        """Diffusion reconstructs target ITEM embedding, not user ID embedding."""
        cond = self._src_condition(users, src_histories, training_mask=True)
        x0 = self.emb_item_tgt(pos_items)
        t = torch.randint(self.diff_tgt.timesteps, (len(users),), device=users.device)
        loss, _ = self.diff_tgt.p_losses(
            x_start=x0, t=t, cond_src=cond,
            cond_tgt=torch.zeros_like(cond), loss_type='l2')
        return loss

    @torch.no_grad()
    def _ema_prototypes(self, global_users, src_histories):
        """Identical deterministic inference path for overlap and cold users."""
        prev_training = self.src_interaction_agg.training
        self.src_interaction_agg.eval()
        try:
            cond = self._src_condition(global_users, src_histories, training_mask=False)
        finally:
            self.src_interaction_agg.train(prev_training)
        teacher = self.diff_tgt_ema if self.use_ema_for_prototypes else self.diff_tgt
        teacher_was_training = teacher.training
        teacher.eval()
        try:
            return teacher.sample_from_noise(
                cond_src=cond, cond_tgt=torch.zeros_like(cond))
        finally:
            teacher.train(teacher_was_training)

    def _refiner_negatives(self, users, pos_items):
        """Uniform sampled negatives, excluding each overlap user's known positives."""
        B, n = len(users), self.refiner_num_neg
        out = torch.empty((B, n), dtype=torch.long, device=users.device)
        for row, (uid, pos) in enumerate(zip(users.tolist(), pos_items.tolist())):
            forbidden = self.overlap_target_positives.get(uid, set()) | {pos}
            if len(forbidden) >= self.num_items_tgt:
                raise ValueError(f'No available negative target items for overlap user {uid}')
            sampled = []
            while len(sampled) < n:
                candidate = torch.randint(1, self.num_items_tgt + 1,
                                          (max(2 * (n - len(sampled)), 8),),
                                          device=users.device).tolist()
                sampled.extend(x for x in candidate if x not in forbidden)
            out[row] = torch.tensor(sampled[:n], dtype=torch.long, device=users.device)
        return out

    def _refiner_softmax_loss(self, overlap_users, pos_items):
        """Only refiner parameters receive gradients from this loss."""
        # EMA model and source/history encoder both frozen in this branch.
        with torch.no_grad():
            hist = self.history_src_user_src[overlap_users]
            proto = self._ema_prototypes(overlap_users, hist)
            negatives = self._refiner_negatives(overlap_users, pos_items)
            positive_emb = self.emb_item_tgt(pos_items).detach()
            negative_emb = self.emb_item_tgt(negatives).detach()

        refined = self.prototype_refiner(proto.detach())
        pos_logits = (refined * positive_emb).sum(dim=-1, keepdim=True)
        neg_logits = (refined.unsqueeze(1) * negative_emb).sum(dim=-1)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.refiner_temperature
        # First candidate is always the real target-domain positive.
        return F.cross_entropy(logits, torch.zeros(len(overlap_users),
                                                  dtype=torch.long, device=overlap_users.device))

    def pre_epoch_processing(self, epoch_idx):
        # Kept for existing trainer hooks. EMA is already initialized in __init__.
        self.current_epoch = epoch_idx

    @torch.no_grad()
    def _momentum_update(self):
        m = float(self.config.get('ema_momentum', 0.999))
        for p, p_ema in zip(self.diff_tgt.parameters(), self.diff_tgt_ema.parameters()):
            p_ema.lerp_(p, 1.0 - m)

    def post_batch_processing(self, epoch_idx, batch_idx):
        """CALL THIS ONLY AFTER optimizer.step(). Never update EMA in forward."""
        if self.use_ema_for_prototypes:
            self._momentum_update()
        self.global_step += 1
        if (self.use_iterative_refinement
                and epoch_idx >= self.pseudo_start_epoch and self.pseudo_update_interval > 0
                and self.global_step % self.pseudo_update_interval == 0):
            self.rebuild_src_only_pseudo_with_ema()

    @torch.no_grad()
    def rebuild_src_only_pseudo_with_ema(self):
        """EMA prototype -> collaborative refiner -> full-catalog Top-K update."""
        top_k = min(int(self.config['top_k_pos']), self.num_items_tgt)
        start, end = self.num_users_overlap + 1, self.num_users_src + 1
        new_pairs = []
        new_pos = {}
        self.prototype_refiner.eval()
        target_emb = self.emb_item_tgt.weight[1:].detach()
        try:
            for s in range(start, end, self.pseudo_generation_batch_size):
                e = min(s + self.pseudo_generation_batch_size, end)
                users = torch.arange(s, e, device=self.emb_user.weight.device)
                hist = self.history_src_user_src[users]
                proto = self._ema_prototypes(users, hist)
                adjusted = self.prototype_refiner(proto) if self.use_collaborative_refiner else proto
                idx = (adjusted @ target_emb.T).topk(k=top_k, dim=1).indices + 1
                for uid, item_row in zip(users.tolist(), idx.tolist()):
                    new_pos[uid] = set(item_row)
                    new_pairs.extend((uid, it) for it in item_row)
        finally:
            self.prototype_refiner.train(self.training)
        self.src_only_pseudo_pos = new_pos
        self.pseudo_ui_pairs = new_pairs
        self.pseudo_refinement_count += 1

    def calculate_loss(self, interaction, epoch_idx):
        # Real overlap target pairs: exclude target-only users; no held-out pairs
        # may appear in interaction['pos_items_tgt'] (enforced upstream).
        users_tgt = interaction['users_tgt'].long()
        pos_tgt = interaction['pos_items_tgt'].long()
        overlap_mask = (users_tgt > 0) & (users_tgt <= self.num_users_overlap)
        overlap_users = users_tgt[overlap_mask]
        overlap_pos = pos_tgt[overlap_mask]

        zero = self.emb_user.weight.sum() * 0.0
        loss_real = zero
        if overlap_users.numel():
            loss_real = self._target_reconstruction(
                overlap_users, overlap_pos, self.history_src_user_src[overlap_users])

        loss_pseudo = zero
        if epoch_idx >= self.pseudo_start_epoch and self.pseudo_ui_pairs:
            B = int(self.config.get('pseudo_num_per_batch', len(overlap_users) or 1))
            if B > 0:
                indices = torch.randint(len(self.pseudo_ui_pairs), (B,), device=self.device)
                pairs = [self.pseudo_ui_pairs[j] for j in indices.tolist()]
                pseudo_u = torch.tensor([u for u, _ in pairs], device=self.device)
                pseudo_i = torch.tensor([i for _, i in pairs], device=self.device)
                loss_pseudo = self._target_reconstruction(
                    pseudo_u, pseudo_i, self.history_src_user_src[pseudo_u])

        loss_ref = zero
        if (epoch_idx >= self.refiner_start_epoch
                and self.use_iterative_refinement and self.use_collaborative_refiner
                and overlap_users.numel() and self.global_step % self.refiner_train_interval == 0):
            loss_ref = self._refiner_softmax_loss(overlap_users, overlap_pos)

        # Disjoint gradient pathways:
        #   Diffusion/conditions/items <- loss_real + loss_pseudo
        #   prototype_refiner         <- loss_ref ONLY
        # Refiner -> Diffusion influence occurs exclusively via pseudo-label updates.
        total = (self.diff_weight * (loss_real + self.pseudo_diff_weight * loss_pseudo)
                 + self.refiner_weight * loss_ref)
        self.last_loss_breakdown = {
            'diff_real': float(loss_real.detach()),
            'diff_pseudo': float(loss_pseudo.detach()),
            'refiner_softmax': float(loss_ref.detach()),
        }
        return total

    @torch.no_grad()
    def full_sort_predict(self, interaction, is_warm):
        users = interaction[0].long()
        if is_warm:
            # Target user IDs of overlap users coincide with source user IDs.
            # Target-only users have no observed source history.
            offset = self.num_users_src - self.num_users_overlap
            global_users = users + (users > self.num_users_overlap).long() * offset
            histories = self.history_tgt_user_src[users]
        else:
            global_users = users
            histories = self.history_src_user_src[users]
        cond = self._src_condition(global_users, histories, training_mask=False)
        prototype = self.diff_tgt.sample_from_noise(
            cond_src=cond, cond_tgt=torch.zeros_like(cond))
        scores = prototype @ self.emb_item_tgt.weight.T
        scores[:, 0] = -1e9
        return scores


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-4, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(
        - beta_min / timesteps
        - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps)
    )
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    """
    a: [T]
    t: [B]  (same device as a)
    return: [B, 1, 1, ...] broadcastable to x_shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = int(config['timesteps'])
        self.beta_start = float(config['beta_start'])
        self.beta_end = float(config['beta_end'])
        self.embedding_dim = int(config['embedding_dim'])
        self.beta_sche = config['beta_sche']

        if self.beta_sche == 'linear':
            betas = linear_beta_schedule(self.timesteps, self.beta_start, self.beta_end)
        elif self.beta_sche == 'exp':
            betas = exp_beta_schedule(self.timesteps)
        elif self.beta_sche == 'cosine':
            betas = cosine_beta_schedule(self.timesteps)
        elif self.beta_sche == 'sqrt':
            betas = betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 1e-4))
        else:
            raise ValueError(f"Unknown beta_sche: {self.beta_sche}")

        self.register_buffer("betas", betas.float())
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)
        self.register_buffer("posterior_variance", posterior_variance)

        self.w_q = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.w_k = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.w_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        init(self.w_q); init(self.w_k); init(self.w_v)
        self.ln = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)

    def get_timestep_embedding(self, timesteps: torch.Tensor, embedding_dim: int):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def selfAttention(self, features: torch.Tensor):
        mask = (features.abs().sum(dim=-1) > 0) # [B, N]

        features = self.ln(features)
        q = self.w_q(features)
        k = self.w_k(features)
        v = self.w_v(features)
        attn_logits = (q * (self.embedding_dim ** -0.5)) @ k.transpose(-1, -2)  # [B, N, N]
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = attn_logits.softmax(dim=-1)  # [B, N, N]
        out = attn @ v  # [B, N, D]
        out = out * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return out.sum(dim=1) / denom

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t = t.to(x_start.device)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond_src=None, cond_tgt=None, loss_type="l2"):
        device = x_start.device
        t = t.to(device)

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)  # [B, D] on device
        tokens = torch.stack([x_noisy, t_emb, cond_src, cond_tgt], dim=1) # [B, 4, D]
        predicted_x0 = self.selfAttention(tokens)

        if loss_type == "l2":
            loss = F.mse_loss(predicted_x0, x_start)
        elif loss_type == "l1":
            loss = F.l1_loss(predicted_x0, x_start)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_x0, x_start)
        else:
            raise NotImplementedError(f"Unknown loss_type: {loss_type}")

        return loss, predicted_x0

    @torch.no_grad()
    def p_sample(self, x_t, t, t_index, cond_src, cond_tgt):
        device = x_t.device
        t = t.to(device)

        t_emb = self.get_timestep_embedding(t, self.embedding_dim)
        tokens = torch.stack([x_t, t_emb, cond_src, cond_tgt], dim=1)
        x_start = self.selfAttention(tokens)

        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean

        var = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_from_noise(self, cond_src, cond_tgt=None):
        """Generate target-domain preference prototype from a Gaussian prior.

        Crucially, generation does NOT require a target item or target user ID
        embedding as x_start; it is usable for source-only cold-start users.
        """
        if cond_tgt is None:
            cond_tgt = torch.zeros_like(cond_src)
        x_t = torch.randn_like(cond_src)
        for n in reversed(range(self.timesteps)):
            t = torch.full((len(cond_src),), n, dtype=torch.long, device=cond_src.device)
            x_t = self.p_sample(x_t=x_t, t=t, t_index=n,
                                cond_src=cond_src, cond_tgt=cond_tgt)
        return x_t

    @torch.no_grad()
    def sample(self, x_start=None, cond_src=None, cond_tgt=None):
        """Compatibility wrapper for older callers; x_start is NOT used."""
        if cond_src is None:
            raise ValueError('cond_src is required')
        result = self.sample_from_noise(cond_src, cond_tgt)
        return x_start, result, None, None, None



class InteractionAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.aggregator = config['aggregator']
        self.embedding_dim = config['embedding_dim']
        dropout_rate = config['dropout']

        self.W_agg = nn.Linear(config['embedding_dim'], config['embedding_dim'], bias=False)

        if self.aggregator in ["user_attention"]:
            self.W_att = nn.Sequential(
                nn.Linear(config['embedding_dim'], config['embedding_dim']),
                nn.Tanh()
            )
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None


    def forward(self, item_emb, user_emb=None):
        """
        item_emb: [B, L, D]
        user_emb: [B, D] (required for user_attention)
        return:   [B, D]
        """
        # padding mask
        mask = (item_emb.abs().sum(dim=-1) > 0)  # [B, L]

        if self.aggregator == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (item_emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            return self.W_agg(pooled)

        elif self.aggregator == "user_attention":
            assert user_emb is not None

            key = self.W_att(item_emb)                     # [B, L, D]
            att = torch.bmm(key, user_emb.unsqueeze(-1))   # [B, L, 1]
            att = att.squeeze(-1)                          # [B, L]
            att = att.masked_fill(~mask, -1e9)
            att = torch.softmax(att, dim=1)

            if self.dropout is not None:
                att = self.dropout(att)

            pooled = torch.bmm(att.unsqueeze(1), item_emb).squeeze(1)  # [B, D]
            return self.W_agg(pooled)

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")




def _extract_item_profile(meta: dict) -> dict:

    if meta is None:
        return {
            "title": "None",
            "description": "None",
            "categories": "None",
            "price": "None",
            "brand": "None",
        }

    # title
    title = meta.get("title")
    title = title if isinstance(title, str) and title.strip() else "None"

    # description
    description = meta.get("description")
    description = description if isinstance(description, str) and description.strip() else "None"

    # categories (flatten + deduplicate)
    raw_categories = meta.get("categories")
    categories = []
    if isinstance(raw_categories, list):
        for path in raw_categories:
            if isinstance(path, list):
                categories.extend([c for c in path if isinstance(c, str) and c.strip()])
            elif isinstance(path, str) and path.strip():
                categories.append(path)
    categories = list(dict.fromkeys(categories))  # deduplicate, keep order
    categories = ", ".join(categories) if categories else "None"

    # price
    price = meta.get("price")
    price = str(price) if isinstance(price, (int, float)) else "None"

    # brand
    brand = meta.get("brand")
    brand = brand if isinstance(brand, str) and brand.strip() else "None"

    return {
        "title": title, 
        "description": description, 
        "categories": categories, 
        "price": price, 
        "brand": brand, 
    }


def _normalize_review(review):

    if isinstance(review, str) and review.strip():
        return review
    return "None"


def _collect_domain_user_items(
    interaction_df,
    domain: str,
    id_mapping: dict,
    metadata: dict,
    reviews: dict,
    user_item_dict: dict,
):
    """
    Collect item profiles + reviews for one domain (src or tgt).
    """
    for row in interaction_df.itertuples(index=False):
        user_id = row.user
        item_id = row.item

        raw_user_id = id_mapping[domain]["id2user"][user_id]
        raw_item_id = id_mapping[domain]["id2item"][item_id]

        if raw_user_id not in user_item_dict:
            user_item_dict[raw_user_id] = {"src": [], "tgt": []}

        meta = metadata[domain].get(raw_item_id)
        review = reviews[domain].get((raw_user_id, raw_item_id))

        item_profile = _extract_item_profile(meta)
        item_profile["review"] = _normalize_review(review)

        user_item_dict[raw_user_id][domain].append(item_profile)


def _collect_domain_items(
    domain: str,
    id_mapping: dict,
    metadata: dict,
    item_profile_dict: dict,
):
    id2item = id_mapping[domain]["id2item"]  # list, index 0 is padding

    # iterate all items in mapping (skip padding idx 0)
    for item_id in range(1, len(id2item)):
        raw_item_id = id2item[item_id]

        if raw_item_id in item_profile_dict:
            continue

        meta = metadata[domain].get(raw_item_id)
        item_profile_dict[raw_item_id] = _extract_item_profile(meta)


def _user_domain_items_to_string(user_profile):
    """
    Convert a user's item-level profile (list[dict[str, str]])
    into a JSON-like string.

    Contract:
    - item is dict[str, str]
    - all values are already strings (including "None")
    - this function only does formatting + minimal escaping
    """
    if not user_profile:
        return "[]"

    ordered_keys = ["title", "description", "categories", "price", "brand", "review"]

    lines = ["["]
    for item in user_profile:
        if not isinstance(item, dict):
            continue

        fields = []
        for k in ordered_keys:
            v = item[k]  #
            v_escaped = v.replace("\\", "\\\\").replace("\"", "\\\"")
            fields.append(f"\"{k}\": \"{v_escaped}\"")

        item_str = "{ " + ", ".join(fields) + " }"
        lines.append(item_str)

    lines.append("]")
    return "\n".join(lines)


def _user_item_profile_to_string(user_profile: dict) -> dict:
    """
    Convert one user's cross-domain item profiles to strings.

    Returns:
        {
            "src": "<string>",
            "tgt": "<string>"
        }
    """
    return {
        "src": _user_domain_items_to_string(user_profile["src"]),
        "tgt": _user_domain_items_to_string(user_profile["tgt"]),
    }


def _build_user_prompt_string(user_profile_string: dict) -> str:
    """
    Build a single prompt string for one user.

    Input:
        user_profile_string = {
            "src": "<JSON-like string>",
            "tgt": "<JSON-like string>"
        }

    Output:
        prompt_str (str)
    """
    return (
        "INTERACTIONS FROM ELECTRONICS:\n"
        f"{user_profile_string['src']}\n\n"
        "INTERACTIONS FROM PHONES:\n"
        f"{user_profile_string['tgt']}"
    )


def _build_all_user_prompt_strings(user_profile_strings: dict) -> dict:
    """
    Build prompt strings for all users.

    Returns:
        { raw_user_id: prompt_string }
    """
    user_prompts = {}

    for user_id, profile_string in user_profile_strings.items():
        user_prompts[user_id] = _build_user_prompt_string(profile_string)

    return user_prompts


def _build_item_prompt_string(item_profile: dict) -> str:
    """
    Build user prompt for ONE item.
    """
    return json.dumps(item_profile, indent=2)


def _extract_string_list(text: str):
    """
    Try to extract a List[str] from LLM output.
    Accepts JSON-style or Python-style lists.
    Returns list[str] or None.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()

    # fast path: JSON list
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, str) for x in obj):
            return obj
    except Exception:
        pass

    # fallback: Python literal list
    try:
        import ast
        obj = ast.literal_eval(text)
        if isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, str) for x in obj):
            return obj
    except Exception:
        pass

    return None


def _process_single_user(args):
    """
    Worker function for one user.
    Each user has at most max_retry attempts.
    """
    user_id, prompt, system_prompt, max_retry = args

    from openai import OpenAI
    client = OpenAI()

    last_response_text = None

    for _ in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                # temperature=0.2,
                # max_tokens=512,
                # extra_body={
                #     "chat_template_kwargs": {"enable_thinking": False},
                # },
            )

            text = response.choices[0].message.content.strip()
            last_response_text = text

            tags = _extract_string_list(text)
            if tags is not None:
                return {
                    "user_id": user_id,
                    "summary": tags,
                    "success": True,
                }

        except Exception as e:
            last_response_text = str(e)

    return {
        "user_id": user_id,
        "summary": last_response_text,
        "success": False,
    }



def _run_user_summarization_multiprocess(
    user_prompts: dict,
    system_prompt: str,
    max_retry: int = 2,
    num_workers: int = None,
):
    """
    Run LLM summarization for all users using multiprocessing.
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    tasks = [
        (user_id, prompt, system_prompt, max_retry)
        for user_id, prompt in user_prompts.items()
    ]

    results = {}
    error_users = {}

    with Pool(processes=num_workers) as pool:
        for out in pool.imap_unordered(_process_single_user, tasks):
            user_id = out["user_id"]
            if out["success"]:
                results[user_id] = out["summary"]
            else:
                error_users[user_id] = out["summary"]

    return results, error_users


def extract_CrossDomain_semantics_both_tag_modality_data(config, modality, interaction, id_mapping, raw_data_list):
    if config['dataset'] == "Amazon2014":
        reviews, metadata = raw_data_list
    elif config['dataset'] == "Douban":
        reviews = raw_data_list[0]
        metadata = raw_data_list[2]

    """
    Step 1:
    Collect per-user cross-domain item profiles + reviews.
    """
  
    user_item_profiles = {}

    _collect_domain_user_items(
        interaction_df=interaction["src"],
        domain="src",
        id_mapping=id_mapping,
        metadata=metadata,
        reviews=reviews,
        user_item_dict=user_item_profiles,
    )
    _collect_domain_user_items(
        interaction_df=interaction["tgt"],
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        reviews=reviews,
        user_item_dict=user_item_profiles,
    )

    
    item_profiles_src = {}
    item_profiles_tgt = {}

    _collect_domain_items(
        domain="src",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_src,
    )

    _collect_domain_items(
        domain="tgt",
        id_mapping=id_mapping,
        metadata=metadata,
        item_profile_dict=item_profiles_tgt,
    )

  
    profile_users = set(user_item_profiles.keys()) 
    src_users = set(id_mapping["src"]["id2user"][1:])  
    tgt_users = set(id_mapping["tgt"]["id2user"][1:])  
    profile_src_users = {u for u, v in user_item_profiles.items() if len(v["src"]) > 0} 
    profile_tgt_users = {u for u, v in user_item_profiles.items() if len(v["tgt"]) > 0} 
    mapping_overlap_users = src_users & tgt_users 
    profile_overlap_users = profile_src_users & profile_tgt_users
    assert profile_users.issubset(src_users | tgt_users), \
        "user_item_profiles contains users not in src or tgt id_mapping"
    assert profile_src_users == src_users, \
        f"Mismatch in src users: profile={len(profile_src_users)}, mapping={len(src_users)}"
    assert profile_tgt_users == tgt_users, \
        f"Mismatch in tgt users: profile={len(profile_tgt_users)}, mapping={len(tgt_users)}"
    assert profile_overlap_users == mapping_overlap_users, \
        f"Mismatch in overlap users: profile={len(profile_overlap_users)}, mapping={len(mapping_overlap_users)}"

    
    src_items = set(id_mapping["src"]["id2item"][1:])
    tgt_items = set(id_mapping["tgt"]["id2item"][1:])
    profile_src_items = set(item_profiles_src.keys())
    profile_tgt_items = set(item_profiles_tgt.keys())
    assert profile_src_items == src_items, \
        f"Mismatch in src items: profile={len(profile_src_items)}, mapping={len(src_items)}"
    assert profile_tgt_items == tgt_items, \
        f"Mismatch in tgt items: profile={len(profile_tgt_items)}, mapping={len(tgt_items)}"

    
    user_profile_strings = {
        user_id: _user_item_profile_to_string(profile)
        for user_id, profile in user_item_profiles.items()
    }

    
    item_prompts_src = {
        item_id: _build_item_prompt_string(profile)
        for item_id, profile in item_profiles_src.items()
    }

    
    item_prompts_tgt = {
        item_id: _build_item_prompt_string(profile)
        for item_id, profile in item_profiles_tgt.items()
    }

    user_prompts = _build_all_user_prompt_strings(user_profile_strings)

    """
    Step 2:
    call LLM to summarize users (multiprocess).
    """
    import os
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    os.environ["OPENAI_BASE_URL"] = config['openai_base_url']

    
    results, error_users = _run_user_summarization_multiprocess(
        user_prompts=user_prompts,
        system_prompt=CROSSDOMAIN_USER_SYSTEM_PROMPT,
        max_retry=2,
        num_workers=20,
    )

    
    data_items_src, error_items_src = _run_user_summarization_multiprocess(
        user_prompts=item_prompts_src,
        system_prompt=SRC_ITEM_SYSTEM_PROMPT,
        max_retry=2,
        num_workers=20,
    )

   
    data_items_tgt, error_items_tgt = _run_user_summarization_multiprocess(
        user_prompts=item_prompts_tgt,
        system_prompt=TGT_ITEM_SYSTEM_PROMPT,
        max_retry=2,
        num_workers=20,
    )

    return {
        "data_users": results,
        "error_users": error_users,
        "data_items_src": data_items_src,
        "error_items_src": error_items_src,
        "data_items_tgt": data_items_tgt,
        "error_items_tgt": error_items_tgt,
    }


def generate_CrossDomain_semantics_both_tag_embs(
    config,
    modality,
    interaction,
    id_mapping,
    modality_data,
):
    """
    Generate cross-domain user semantic embeddings.

    Output shape:
        [num_src_user + num_tgt_user - num_overlap_user, embedding_dim]

    User order:
        1 ~ num_overlap_user                 : overlap users
        num_overlap_user+1 ~ num_src_user    : src-only users
        num_overlap_user+1 ~ num_tgt_user    : tgt-only users
    """
    if len(modality_data["error_users"]):
        raise ValueError(f"There are error users in modality data json file.")
    if len(modality_data["error_items_src"]) > 0:
        raise ValueError("There are error src items in modality data json file.")
    if len(modality_data["error_items_tgt"]) > 0:
        raise ValueError("There are error tgt items in modality data json file.")

    # =========================================================
    # Step 0: infer number of overlap users (prefix-based)
    # =========================================================
    src_id2user = id_mapping["src"]["id2user"]
    tgt_id2user = id_mapping["tgt"]["id2user"]

    num_overlap_user = 0
    max_check = min(len(src_id2user), len(tgt_id2user))
    for i in range(1, max_check):
        if src_id2user[i] == tgt_id2user[i]:
            num_overlap_user += 1
        else:
            break

    num_src_user = len(src_id2user) - 1  # exclude padding idx 0
    num_tgt_user = len(tgt_id2user) - 1

    # =========================================================
    # Step 1: prepare user summaries
    # =========================================================
    # modality_data = {"data": {raw_user_id: summary, ...}, "error_users": {...}}
    user_summaries = modality_data["data_users"]
    item_summaries_src = modality_data["data_items_src"]
    item_summaries_tgt = modality_data["data_items_tgt"]

    embedding_model = modality["emb_model"]
    batch_size = modality["emb_batch_size"]
    normalize = modality.get("normalize_semantic_emb", False)

    # =========================================================
    # Step 2: build ordered user list (CRITICAL)
    # =========================================================
    ordered_raw_users = []
    ordered_tag_lists = []  # List[List[str]]


    # (1) overlap users
    for uid in range(1, num_overlap_user + 1):
        raw_user = src_id2user[uid]
        if raw_user not in user_summaries:
            raise ValueError(f"Missing summary for overlap user {raw_user}")
        ordered_raw_users.append(raw_user)
        ordered_tag_lists.append(user_summaries[raw_user])

    # (2) src-only users
    for uid in range(num_overlap_user + 1, num_src_user + 1):
        raw_user = src_id2user[uid]
        if raw_user not in user_summaries:
            raise ValueError(f"Missing summary for src-only user {raw_user}")
        ordered_raw_users.append(raw_user)
        ordered_tag_lists.append(user_summaries[raw_user])

    # (3) tgt-only users
    for uid in range(num_overlap_user + 1, num_tgt_user + 1):
        raw_user = tgt_id2user[uid]
        if raw_user not in user_summaries:
            raise ValueError(f"Missing summary for tgt-only user {raw_user}")
        ordered_raw_users.append(raw_user)
        ordered_tag_lists.append(user_summaries[raw_user])

    # =========================================================
    # Step 2.5: build ordered item lists (CRITICAL)
    # =========================================================
    # src items
    src_id2item = id_mapping["src"]["id2item"]
    ordered_src_items = []
    ordered_src_item_tags = []

    for iid in range(1, len(src_id2item)):
        raw_item = src_id2item[iid]
        if raw_item not in item_summaries_src:
            raise ValueError(f"Missing summary for src item {raw_item}")
        ordered_src_items.append(raw_item)
        ordered_src_item_tags.append(item_summaries_src[raw_item])

    # tgt items
    tgt_id2item = id_mapping["tgt"]["id2item"]
    ordered_tgt_items = []
    ordered_tgt_item_tags = []

    for iid in range(1, len(tgt_id2item)):
        raw_item = tgt_id2item[iid]
        if raw_item not in item_summaries_tgt:
            raise ValueError(f"Missing summary for tgt item {raw_item}")
        ordered_tgt_items.append(raw_item)
        ordered_tgt_item_tags.append(item_summaries_tgt[raw_item])

    # =========================================================
    # Step 3: batch encode summaries
    # =========================================================
    # user
    flat_texts = []  
    flat_user_idx = []  
    for user_idx, tag_list in enumerate(ordered_tag_lists):
        for tag in tag_list:
            flat_texts.append(tag)
            flat_user_idx.append(user_idx)

    # item
    flat_item_texts = []
    flat_item_idx = []  # index in [src_items + tgt_items]
    # src items first
    for item_idx, tag_list in enumerate(ordered_src_item_tags):
        for tag in tag_list:
            flat_item_texts.append(tag)
            flat_item_idx.append(item_idx)
    src_item_offset = len(ordered_src_items)
    # tgt items
    for item_idx, tag_list in enumerate(ordered_tgt_item_tags):
        for tag in tag_list:
            flat_item_texts.append(tag)
            flat_item_idx.append(src_item_offset + item_idx)

    
    all_flat_texts = flat_texts + flat_item_texts
    num_flat = len(all_flat_texts)
    if num_flat == 0:
        raise ValueError("No tag texts to embed.")

    num_users = len(ordered_raw_users)
    num_items_total = len(ordered_src_items) + len(ordered_tgt_items)
    
    sum_user_embs = None
    cnt_user = np.zeros((num_users,), dtype=np.int32)
    sum_item_embs = None
    cnt_item = np.zeros((num_items_total,), dtype=np.int32)
    num_user_tags = len(flat_texts)  

    # if 'text-embedding-3' in modality['emb_model']:
    #     from openai import OpenAI
    #
    #     client = OpenAI(
    #         api_key=config["openai_api_key"],
    #         base_url=config.get("openai_base_url", None),
    #     )
    #
    #     flat_embs = []
    #     for start in range(0, num_flat, batch_size):
    #         end = min(start + batch_size, num_flat)
    #         batch_texts = all_flat_texts[start:end]
    #
    #         response = client.embeddings.create(
    #             model=embedding_model,
    #             input=batch_texts,
    #         )
    #
    #         for emb_obj in response.data:
    #             emb = np.asarray(emb_obj.embedding, dtype=np.float32)
    #             if normalize:
    #                 emb = emb / (np.linalg.norm(emb) + 1e-12)
    #             flat_embs.append(emb)
    #
    #     flat_embs = np.stack(flat_embs, axis=0)  # [num_tags, D]
    if 'text-embedding-3' in modality['emb_model']:
        from openai import OpenAI

        client = OpenAI(
            api_key=config["openai_api_key"],
            base_url=config.get("openai_base_url", None),
        )

        for start in range(0, num_flat, batch_size):
            end = min(start + batch_size, num_flat)
            batch_texts = all_flat_texts[start:end]

            response = client.embeddings.create(
                model=embedding_model,
                input=batch_texts,
            )

           
            for j, emb_obj in enumerate(response.data):
                emb = np.asarray(emb_obj.embedding, dtype=np.float32)

                if normalize:
                    emb = emb / (np.linalg.norm(emb) + 1e-12)

                
                if sum_user_embs is None:
                    emb_dim = emb.shape[0]
                    sum_user_embs = np.zeros((num_users, emb_dim), dtype=np.float32)
                    sum_item_embs = np.zeros((num_items_total, emb_dim), dtype=np.float32)

                global_idx = start + j  

                # 0 .. num_user_tags-1 是 user tags
                if global_idx < num_user_tags:
                    uidx = flat_user_idx[global_idx]
                    sum_user_embs[uidx] += emb
                    cnt_user[uidx] += 1
                else:
                    
                    k = global_idx - num_user_tags
                    iidx = flat_item_idx[k]  
                    sum_item_embs[iidx] += emb
                    cnt_item[iidx] += 1
    else:
        raise NotImplementedError("Only OpenAI embedding models are handled here.")

   
    # emb_dim = flat_embs.shape[1]
    # num_users = len(ordered_raw_users)
    # num_user_tags = len(flat_texts)
    #
    # sum_embs = np.zeros((num_users, emb_dim), dtype=np.float32)
    # cnt = np.zeros((num_users,), dtype=np.int32)
    #
    # for i in range(num_user_tags):
    #     uidx = flat_user_idx[i]
    #     sum_embs[uidx] += flat_embs[i]
    #     cnt[uidx] += 1
    #
    # raw_user_to_emb = {}
    #
    # for uidx, raw_user in enumerate(ordered_raw_users):
    #     user_emb = sum_embs[uidx] / cnt[uidx]
    #     raw_user_to_emb[raw_user] = user_emb
    #

    # num_items_total = len(ordered_src_items) + len(ordered_tgt_items)
    #
    # sum_item_embs = np.zeros((num_items_total, emb_dim), dtype=np.float32)
    # cnt_item = np.zeros((num_items_total,), dtype=np.int32)
    #
    # item_offset = num_user_tags  # item embeddings start after user tags
    #
    # for i in range(len(flat_item_texts)):
    #     idx = flat_item_idx[i]
    #     sum_item_embs[idx] += flat_embs[item_offset + i]
    #     cnt_item[idx] += 1
    #
    # item_embs = sum_item_embs / cnt_item[:, None]
    #
    # # Sanity check
    # assert len(raw_user_to_emb) == len(ordered_raw_users), \
    #     "Some user embeddings are missing after batch encoding"

    if sum_user_embs is None:
        raise ValueError("Embedding failed: sum_user_embs was not initialized.")

   
    cnt_user_safe = np.clip(cnt_user, 1, None).astype(np.float32)
    cnt_item_safe = np.clip(cnt_item, 1, None).astype(np.float32)

    user_embeddings = sum_user_embs / cnt_user_safe[:, None]
    item_embs = sum_item_embs / cnt_item_safe[:, None]

    # =========================================================
    # Step 4: stack embeddings in final order
    # =========================================================
    # user_embeddings = np.stack(
    #     [raw_user_to_emb[raw_user] for raw_user in ordered_raw_users],
    #     axis=0
    # )

    final_embeddings = np.concatenate(
        [
            user_embeddings,
            item_embs[:len(ordered_src_items)],  # src items
            item_embs[len(ordered_src_items):],  # tgt items
        ],
        axis=0,
    )

    return final_embeddings



def generate_CrossDomain_semantics_both_tag_final_embs(config, modality, interaction, id_mapping, modality_embs):
    input_dim = modality_embs.shape[1]
    target_dim = modality["emb_pca"]

    if input_dim == target_dim:
        return modality_embs.astype(np.float32)

    from sklearn.decomposition import PCA
    pca = PCA(
        n_components=target_dim,
        random_state=config.get("seed", 999),
    )
    final_embs = pca.fit_transform(modality_embs)

    if modality["normalize_semantic_emb"]:
        norm = np.linalg.norm(final_embs, axis=1, keepdims=True) + 1e-12
        final_embs = final_embs / norm

    return final_embs.astype(np.float32)



CROSSDOMAIN_USER_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Your task is to summarize a user's interests based on their interactions with items from two different types of product categories.
One category is about electronics.
The other category is about cell phones and accessories.

The information I will give you:
INTERACTIONS FROM ELECTRONICS: A LIST of user interactions with items related to electronics.
INTERACTIONS FROM PHONES: A LIST of user interactions with items related to cell phones and accessories.

Each interaction is described in JSON format with the following attributes, where missing values are set to "None".
The attributes include the item's information and the user's review on that item:
{
  "title": "the name of the item"
  "description": "a description of the item"
  "categories": "several tags describing the item"
  "price": "the price of the item"
  "brand": "the brand of the item"
  "review": "the user's review on the item"
}

Requirements:
1. Extract a set of high-level, abstract user preference tags from the user's interactions.
   The tags should reflect:
   - emotional attitudes,
   - value orientations,
   - comfort or reliability expectations,
   - lifestyle or usage preferences,
   as inferred from item attributes and user reviews.
   Do NOT describe:
   - specific products,
   - item categories,
   - functions,
   - materials,
   - usage scenarios,
   or any concrete physical objects.
2. Output only a list of tags, following this structure: ["tag1", "tag2", "tag3", "..."]
    - tags must be: typically 1–2 words (at most 3)
    - the number of tags should be 2-4
3. Do not provide any other text outside the list.
"""



SRC_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Your task is to analyze ONE product item related to electronics,
and summarize what types of users this item is likely to attract.

The information I will give you is the item's metadata in JSON format,
where any missing field is set to "None":
{
  "title": "item name",
  "description": "item description",
  "categories": "category tags",
  "price": "price",
  "brand": "brand"
}

Requirements:
1. Infer high-level user preference tags that describe what kinds of users may like this item.
   Tags should reflect abstract user tendencies such as:
   - style orientation,
   - comfort or quality expectations,
   - value sensitivity,
   - lifestyle or aesthetic preferences.
   Do NOT mention:
   - specific product names,
   - brands,
   - materials,
   - item categories,
   - concrete usage scenarios.
2. Output ONLY a list of tags, following this format: ["tag1", "tag2", "tag3", "..."]
   - tags must be: typically 1–2 words (at most 3)
   - the number of tags should be 2–4
3. Do not provide any other text outside the list.
"""



TGT_ITEM_SYSTEM_PROMPT = """
You are an expert in recommendation systems.
Your task is to analyze ONE product item related to cell phones and accessories,
and summarize what types of users this item is likely to attract.

The information I will give you is the item's metadata in JSON format,
where any missing field is set to "None":
{
  "title": "item name",
  "description": "item description",
  "categories": "category tags",
  "price": "price",
  "brand": "brand"
}

Requirements:
1. Infer high-level user preference tags that describe what kinds of users may like this item.
   Tags should reflect abstract user tendencies such as:
   - performance orientation,
   - durability or reliability expectations,
   - activity intensity preferences,
   - outdoor or fitness lifestyle traits.
   Do NOT mention:
   - specific product names,
   - brands,
   - materials,
   - item categories,
   - concrete usage scenarios.
2. Output ONLY a list of tags, following this format: ["tag1", "tag2", "tag3", "..."]
   - tags must be: typically 1–2 words (at most 3)
   - the number of tags should be 2–4
3. Do not provide any other text outside the list.
"""




