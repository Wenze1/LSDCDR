import os
import json
import numpy as np
import pandas as pd
from typing import List
from logging import getLogger
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.data_preprocessing.amazon_data_processor import AmazonDataProcessor
from utils.data_preprocessing.amazon_modality_processor import AmazonModalityProcessor
from utils.data_preprocessing.douban_data_processor import DoubanDataProcessor
from utils.data_preprocessing.douban_modality_processor import DoubanModalityProcessor

logger = getLogger()


def load_all_item_seqs(src_processed_dir, tgt_processed_dir, shuffle):
    with open(
        os.path.join(
            src_processed_dir,
            f'all_item_seqs_{"shuffle" if shuffle else "noshuffle"}.json',
        ),
        "r",
    ) as f:
        src_all_item_seqs = json.load(f)
    with open(
        os.path.join(
            tgt_processed_dir,
            f'all_item_seqs_{"shuffle" if shuffle else "noshuffle"}.json',
        ),
        "r",
    ) as f:
        tgt_all_item_seqs = json.load(f)
    all_item_seqs = {"src": src_all_item_seqs, "tgt": tgt_all_item_seqs}
    return all_item_seqs


def split_users_and_reindex(config, joint_path, all_item_seqs):

    all_users_file_path = os.path.join(joint_path, "all_users.json")
    id_mapping_file_path = os.path.join(joint_path, "id_mapping.json")
    if os.path.exists(all_users_file_path) and os.path.exists(id_mapping_file_path):
        logger.info("[DATASET] User splitting and reindex have been done...")
        with open(all_users_file_path, "r") as f:
            all_users = json.load(f)
        with open(id_mapping_file_path, "r") as f:
            id_mapping = json.load(f)
        return all_users, id_mapping

    users_src = sorted(all_item_seqs["src"].keys())
    users_tgt = sorted(all_item_seqs["tgt"].keys())

    users_src_set = set(users_src)
    users_tgt_set = set(users_tgt)
    overlap_users = sorted(users_src_set & users_tgt_set)
    src_only_users = sorted(users_src_set - set(overlap_users))
    tgt_only_users = sorted(users_tgt_set - set(overlap_users))

    rng = np.random.RandomState(999)
    overlap_list = rng.permutation(overlap_users)

    num_overlap = len(overlap_list)
    num_valid_cold = int(num_overlap * config["t_cold_valid"])
    num_test_cold = int(num_overlap * config["t_cold_test"])
    assert (
        num_valid_cold + num_test_cold <= num_overlap
    ), "Sum of t_cold_valid and t_cold_test causes user overflow."
    valid_cold_users = sorted(overlap_list[:num_valid_cold])
    test_cold_users = sorted(
        overlap_list[num_valid_cold : num_valid_cold + num_test_cold]
    )
    overlap_users = sorted(overlap_list[num_valid_cold + num_test_cold :])

    all_users = {
        "overlap_users": overlap_users,
        "valid_cold_users": valid_cold_users,
        "test_cold_users": test_cold_users,
        "src_only_users": src_only_users,
        "tgt_only_users": tgt_only_users,
    }

    id_mapping = {
        "src": {
            "user2id": {},
            "item2id": {},
            "id2user": ["[PAD]"],
            "id2item": ["[PAD]"],
        },
        "tgt": {
            "user2id": {},
            "item2id": {},
            "id2user": ["[PAD]"],
            "id2item": ["[PAD]"],
        },
    }

    for i, u in enumerate(all_users["overlap_users"], start=1):
        id_mapping["src"]["user2id"][u] = i
        id_mapping["src"]["id2user"].append(u)
        id_mapping["tgt"]["user2id"][u] = i
        id_mapping["tgt"]["id2user"].append(u)
    src_extra_users = (
        all_users["valid_cold_users"]
        + all_users["test_cold_users"]
        + all_users["src_only_users"]
    )

    for i, u in enumerate(src_extra_users, start=len(all_users["overlap_users"]) + 1):
        id_mapping["src"]["user2id"][u] = i
        id_mapping["src"]["id2user"].append(u)

    for i, u in enumerate(
        all_users["tgt_only_users"], start=len(all_users["overlap_users"]) + 1
    ):
        id_mapping["tgt"]["user2id"][u] = i
        id_mapping["tgt"]["id2user"].append(u)

    for domain in ["src", "tgt"]:
        all_items = set()
        for items in all_item_seqs[domain].values():
            all_items.update(items)
        for item in sorted(all_items):
            id_mapping[domain]["item2id"][item] = len(id_mapping[domain]["id2item"])
            id_mapping[domain]["id2item"].append(item)

    all_users = {
        "overlap_users": [
            id_mapping["src"]["user2id"][u] for u in all_users["overlap_users"]
        ],
        "valid_cold_users": [
            id_mapping["src"]["user2id"][u] for u in all_users["valid_cold_users"]
        ],
        "test_cold_users": [
            id_mapping["src"]["user2id"][u] for u in all_users["test_cold_users"]
        ],
        "src_only_users": [
            id_mapping["src"]["user2id"][u] for u in all_users["src_only_users"]
        ],
        "tgt_only_users": [
            id_mapping["tgt"]["user2id"][u] for u in all_users["tgt_only_users"]
        ],
    }

    logger.info("[DATASET] Saving user splitting and id mapping...")
    with open(all_users_file_path, "w") as f:
        json.dump(all_users, f)
    with open(id_mapping_file_path, "w") as f:
        json.dump(id_mapping, f)
    return all_users, id_mapping


def filter_overlap_users(all_item_seqs, k, joint_dataset_name):
    """Implementation details."""

    def dual_domain_kcore_loop(src, tgt, k):
        prev_src, prev_tgt = None, None
        iteration = 0
        while prev_src != src or prev_tgt != tgt:
            iteration += 1
            prev_src, prev_tgt = src, tgt
            overlap_users = [u for u in src.keys() if u in tgt]
            src = {u: seq for u, seq in src.items() if u in overlap_users}
            tgt = {u: seq for u, seq in tgt.items() if u in overlap_users}
            src = {u: seq for u, seq in src.items() if len(seq) >= k}
            tgt = {u: seq for u, seq in tgt.items() if len(seq) >= k}

            def filter_items(data):
                item_count = defaultdict(int)
                for seq in data.values():
                    for i in seq:
                        item_count[i] += 1
                valid_items = set(i for i, c in item_count.items() if c >= k)
                return {
                    u: [i for i in seq if i in valid_items] for u, seq in data.items()
                }

            src = filter_items(src)
            tgt = filter_items(tgt)
            src = {u: seq for u, seq in src.items() if len(seq) > 0}
            tgt = {u: seq for u, seq in tgt.items() if len(seq) > 0}
            logger.info(
                f"[JOINT] Iter {iteration}: {len(src)} overlapping users remaining"
            )
        return src, tgt

    all_item_seqs["src"], all_item_seqs["tgt"] = dual_domain_kcore_loop(
        all_item_seqs["src"], all_item_seqs["tgt"], k
    )
    if not len(all_item_seqs["src"]):
        logger.error(
            f"[JOINT] K-Core filtering resulted in **zero users** for dataset {joint_dataset_name} "
            f"with k={k}. Try lowering k_core or use all_users mode."
        )
        raise ValueError(
            f"K-Core filtering failed: no overlapping users remain for {joint_dataset_name}."
        )
    return all_item_seqs


def to_df(inter):
    df = pd.DataFrame(inter, columns=["user", "item"])
    df = df.sort_values(["user", "item"]).reset_index(drop=True)
    return df


def split_interation(config, joint_path, all_item_seqs, all_users, id_mapping):
    train_src_path = os.path.join(joint_path, "train_src.pkl")
    train_tgt_path = os.path.join(joint_path, "train_tgt.pkl")
    valid_cold_tgt_path = os.path.join(joint_path, "valid_cold_tgt.pkl")
    test_cold_tgt_path = os.path.join(joint_path, "test_cold_tgt.pkl")
    valid_warm_tgt_path = os.path.join(joint_path, "valid_warm_tgt.pkl")
    test_warm_tgt_path = os.path.join(joint_path, "test_warm_tgt.pkl")
    if (
        os.path.exists(train_src_path)
        and os.path.exists(train_tgt_path)
        and os.path.exists(valid_cold_tgt_path)
        and os.path.exists(test_cold_tgt_path)
        and os.path.exists(valid_warm_tgt_path)
        and os.path.exists(test_warm_tgt_path)
    ):
        logger.info("[DATASET] Interaction Splitting has been done...")
        train_src = pd.read_pickle(train_src_path)
        train_tgt = pd.read_pickle(train_tgt_path)
        valid_cold_tgt = pd.read_pickle(valid_cold_tgt_path)
        test_cold_tgt = pd.read_pickle(test_cold_tgt_path)
        valid_warm_tgt = pd.read_pickle(valid_warm_tgt_path)
        test_warm_tgt = pd.read_pickle(test_warm_tgt_path)
        return (
            train_src,
            train_tgt,
            valid_cold_tgt,
            test_cold_tgt,
            valid_warm_tgt,
            test_warm_tgt,
        )

    train_src, train_tgt = [], []
    valid_cold_tgt, test_cold_tgt = [], []
    valid_warm_tgt, test_warm_tgt = [], []

    for raw_uid in sorted(all_item_seqs["src"]):
        item_seq = all_item_seqs["src"][raw_uid]
        uid = id_mapping["src"]["user2id"][raw_uid]
        for raw_iid in item_seq:
            iid = id_mapping["src"]["item2id"][raw_iid]
            train_src.append([uid, iid])

    valid_cold_raw_users = {
        id_mapping["src"]["id2user"][uid] for uid in all_users["valid_cold_users"]
    }
    test_cold_raw_users = {
        id_mapping["src"]["id2user"][uid] for uid in all_users["test_cold_users"]
    }
    for raw_uid in sorted(all_item_seqs["tgt"]):
        item_seq = all_item_seqs["tgt"][raw_uid]

        if raw_uid in valid_cold_raw_users:

            uid = id_mapping["src"]["user2id"][raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping["tgt"]["item2id"][raw_iid]
                valid_cold_tgt.append([uid, iid])
        elif raw_uid in test_cold_raw_users:

            uid = id_mapping["src"]["user2id"][raw_uid]
            for raw_iid in item_seq:
                iid = id_mapping["tgt"]["item2id"][raw_iid]
                test_cold_tgt.append([uid, iid])
        elif raw_uid in id_mapping["tgt"]["user2id"]:
            # overlap_users/tgt_only_users

            uid = id_mapping["tgt"]["user2id"][raw_uid]
            seq_len = len(item_seq)
            if seq_len <= 2:
                num_train, num_valid, num_test = seq_len, 0, 0
            else:
                num_test = (
                    max(1, int(seq_len * config["warm_test_ratio"]))
                    if config["warm_test_ratio"]
                    else 0
                )
                num_valid = (
                    max(1, int(seq_len * config["warm_valid_ratio"]))
                    if config["warm_valid_ratio"]
                    else 0
                )
            num_train = seq_len - num_valid - num_test
            assert num_train >= 0
            for raw_iid in item_seq[:num_train]:
                train_tgt.append([uid, id_mapping["tgt"]["item2id"][raw_iid]])
            for raw_iid in item_seq[num_train : num_train + num_valid]:
                valid_warm_tgt.append([uid, id_mapping["tgt"]["item2id"][raw_iid]])
            for raw_iid in item_seq[num_train + num_valid :]:
                test_warm_tgt.append([uid, id_mapping["tgt"]["item2id"][raw_iid]])

    train_src = to_df(train_src)
    train_tgt = to_df(train_tgt)
    valid_cold_tgt = to_df(valid_cold_tgt)
    test_cold_tgt = to_df(test_cold_tgt)
    valid_warm_tgt = to_df(valid_warm_tgt)
    test_warm_tgt = to_df(test_warm_tgt)

    train_src.to_pickle(train_src_path)
    train_tgt.to_pickle(train_tgt_path)
    valid_cold_tgt.to_pickle(valid_cold_tgt_path)
    test_cold_tgt.to_pickle(test_cold_tgt_path)
    valid_warm_tgt.to_pickle(valid_warm_tgt_path)
    test_warm_tgt.to_pickle(test_warm_tgt_path)
    logger.info("[DATASET] Interaction Splitting finished and saved.")

    return (
        train_src,
        train_tgt,
        valid_cold_tgt,
        test_cold_tgt,
        valid_warm_tgt,
        test_warm_tgt,
    )


def prepare_modality_emb(config, domains, id_mapping, train_src, train_tgt, joint_path):
    all_id_mapping = {
        "src": id_mapping["src"],
        "tgt": id_mapping["tgt"],
    }
    train_df = {
        "src": train_src,
        "tgt": train_tgt,
    }

    logger.info(f"[TRAINING] Start joint modality preparation for domains: {domains}")

    try:
        processor_classes = {
            "Amazon2014": AmazonModalityProcessor,
            "Douban": DoubanModalityProcessor,
        }
        processor_class = processor_classes.get(config["dataset"])
        if processor_class is None:
            raise ValueError(f'Unsupported dataset: {config["dataset"]}')

        processor = processor_class(
            config=config,
            domains=domains,
            id_mapping=all_id_mapping,
            train_df=train_df,
            joint_path=joint_path,
        )

        result = processor.run_full_pipeline()

        logger.info(
            "[TRAINING] Joint modality preparation "
            f"{'completed' if result else 'failed'}"
        )

        return result

    except Exception as e:
        logger.error(
            "[ERROR] Exception during joint modality preparation", exc_info=True
        )
        return False


def create_joint_dataset(domains: List[str], config: dict):

    data_path = config["data_path"] if "data_path" in config else "../data/"
    joint_dataset_name = "+".join(domains)
    dataset_type = "only_overlap_users" if config["only_overlap_users"] else "all_users"
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config["only_overlap_users"]:
        split_dir += f'_{config["k_cores"]}cores'
    joint_path = os.path.join(
        data_path, config["dataset"], joint_dataset_name, dataset_type, split_dir
    )

    os.makedirs(os.path.join(joint_path, "modality_emb"), exist_ok=True)
    logger.info(f"[JOINT] Creating joint dataset: {joint_dataset_name}")

    logger.info("\n=== STEP 1: Load all_item_seqs from src & tgt ===")
    src_processed_dir = os.path.join(
        data_path, config["dataset"], domains[0], "processed"
    )
    tgt_processed_dir = os.path.join(
        data_path, config["dataset"], domains[1], "processed"
    )
    all_item_seqs = load_all_item_seqs(
        src_processed_dir, tgt_processed_dir, config["shuffle_user_sequence"]
    )
    if config["only_overlap_users"]:
        all_item_seqs = filter_overlap_users(
            all_item_seqs, config["k_cores"], joint_dataset_name
        )

    # {
    #     "overlap_users": [1, 2, 3, ...],
    #     "valid_cold_users": [3131, 3136, 3222, ...],
    #     "test_cold_users": [3132, 3138, 3199, ...],
    #     "src_only_users": [3129, 3130, 3133, ...],
    #     "tgt_only_users":[3129, 3130, 3131, ...],
    # }

    # {
    #     "src": {
    #         "user2id": {"raw_user_id": 1, "...": "..."},
    #         "id2user": ["[PAD]", "raw_user_id", "..."],
    #         "item2id": {"raw_item_id": 1, "...": "..."},
    #         "id2item": ["[PAD]", "raw_item_id", "..."]
    #     },
    #     "tgt": {
    #         "user2id": {"raw_user_id": 1, "...": "..."},
    #         "id2user": ["[PAD]", "..."],
    #         "item2id": {"raw_item_id": 1, "...": "..."},
    #         "id2item": ["[PAD]", "..."]
    #     }
    # }

    logger.info("\n=== STEP 2: Split users and reindex ===")
    all_users, id_mapping = split_users_and_reindex(config, joint_path, all_item_seqs)

    logger.info("\n=== STEP 3: Split train/valid/test ===")
    (
        train_src,
        train_tgt,
        valid_cold_tgt,
        test_cold_tgt,
        valid_warm_tgt,
        test_warm_tgt,
    ) = split_interation(config, joint_path, all_item_seqs, all_users, id_mapping)

    # modality_emb /

    logger.info("\n=== STEP 4: Prepare modality embeddings ===")
    prepare_modality_emb(config, domains, id_mapping, train_src, train_tgt, joint_path)

    logger.info(f"\n[JOINT] All joint dataset files created successfully!")
    return joint_path


def check_and_prepare_Amazon2014_single(config, domain):

    domain_path = os.path.join(config["data_path"], config["dataset"], domain)
    processed_dir = os.path.join(domain_path, "processed")
    seq_file = (
        f"all_item_seqs_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}.json"
    )

    required_files = [
        os.path.join(processed_dir, seq_file),
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        logger.info(
            f"[TRAINING] [{domain}] Missing files detected, starting data processing pipeline..."
        )
        logger.info(f"[TRAINING] [{domain}] Missing files: {missing_files}")

        try:
            processor = AmazonDataProcessor(config, domain, config["data_path"])
            processor.run_full_pipeline()
            logger.info(f"[TRAINING] [{domain}] Data processing pipeline completed")

            still_missing = [f for f in required_files if not os.path.exists(f)]
            if still_missing:
                logger.error(
                    f"[ERROR] [{domain}] Files still missing after processing: {still_missing}"
                )
                return False

        except Exception as e:
            logger.error(
                f"[ERROR] [{domain}] Error during data processing: {e}", exc_info=True
            )
            return False
    else:
        logger.info(
            f"[TRAINING] [{domain}] All required data files exist, skipping data processing"
        )

    return True


def check_and_prepare_Amazon2014(config):

    domains = config["domains"]

    # Amazon2014 /

    # STEP 1: Checking and preparing individual Amazon datasets
    logger.info(
        f"[TRAINING] Starting parallel data preparation for dataset: Amazon2014, processing domains individually: {domains}"
    )
    with ThreadPoolExecutor(max_workers=len(domains)) as executor:

        future_to_dataset = {
            executor.submit(check_and_prepare_Amazon2014_single, config, domain): domain
            for domain in domains
        }

        results = {}
        for future in as_completed(future_to_dataset):
            domain_name = future_to_dataset[future]
            try:
                result = future.result()
                results[domain_name] = result
                logger.info(
                    f"[TRAINING] [{domain_name}] Data preparation {'completed' if result else 'failed'}"
                )
            except Exception as e:
                logger.error(
                    f"[ERROR] [{domain_name}] Exception during data preparation: {e}",
                    exc_info=True,
                )
                results[domain_name] = False

    failed_domains = [name for name, success in results.items() if not success]
    if failed_domains:
        logger.error(f"[ERROR] Failed to prepare data for domains: {failed_domains}")
        return False
    logger.info(
        f"[TRAINING] All single-domain datasets are prepared successfully: {list(results.keys())}"
    )

    # Amazon2014 /

    # STEP 2: Checking and creating joint Amazon dataset
    if len(domains) != 2:
        logger.info("[TRAINING] Single dataset mode, skipping joint dataset creation")
        return True

    data_path = config["data_path"] if "data_path" in config else "../data/"
    joint_dataset_name = "+".join(domains)
    dataset_type = "only_overlap_users" if config["only_overlap_users"] else "all_users"
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config["only_overlap_users"]:
        split_dir += f'_{config["k_cores"]}cores'
    joint_path = os.path.join(
        data_path, "Amazon2014", joint_dataset_name, dataset_type, split_dir
    )

    required_joint_files = [
        os.path.join(joint_path, "all_users.json"),
        os.path.join(joint_path, "id_mapping.json"),
        os.path.join(joint_path, "train_src.pkl"),
        os.path.join(joint_path, "train_tgt.pkl"),
        os.path.join(joint_path, "valid_cold_tgt.pkl"),
        os.path.join(joint_path, "test_cold_tgt.pkl"),
        os.path.join(joint_path, "valid_warm_tgt.pkl"),
        os.path.join(joint_path, "test_warm_tgt.pkl"),
    ]

    for modality in config["modalities"]:
        if not modality["enabled"]:
            continue
        final_embs_file_name = (
            modality["name"] + "_final_emb_" + str(modality["emb_pca"]) + ".npy"
        )
        required_joint_files.append(
            os.path.join(joint_path, "modality_emb", final_embs_file_name)
        )

    missing_joint_files = [f for f in required_joint_files if not os.path.exists(f)]

    if missing_joint_files:
        logger.info(
            f"[TRAINING] One or more joint dataset files are missing. Recreating ALL joint files."
        )
        logger.info(f"Missing files: {missing_joint_files}")
        try:

            create_joint_dataset(domains=domains, config=config)

            still_missing = [f for f in required_joint_files if not os.path.exists(f)]
            if still_missing:
                logger.error(
                    f"[ERROR] Joint dataset creation failed, still missing: {still_missing}"
                )
                return False
        except Exception as e:
            logger.error(f"[ERROR] Failed to create joint dataset: {e}", exc_info=True)
            return False
    else:
        logger.info(
            f"[TRAINING] All required joint dataset files (including embeddings) already exist. Skipping creation."
        )

    return True


def check_and_prepare_Douban_single(config, domain):
    domain_path = os.path.join(config["data_path"], config["dataset"], domain)
    processed_dir = os.path.join(domain_path, "processed")
    seq_file = (
        f"all_item_seqs_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}.json"
    )
    required_files = [os.path.join(processed_dir, seq_file)]
    missing_files = [path for path in required_files if not os.path.exists(path)]

    if not missing_files:
        logger.info(
            f"[TRAINING] [{domain}] All required data files exist, "
            "skipping data processing"
        )
        return True

    logger.info(
        f"[TRAINING] [{domain}] Missing files detected, starting data "
        "processing pipeline..."
    )
    logger.info(f"[TRAINING] [{domain}] Missing files: {missing_files}")
    try:
        processor = DoubanDataProcessor(config, domain, config["data_path"])
        processor.run_full_pipeline()
    except Exception as error:
        logger.error(
            f"[ERROR] [{domain}] Error during data processing: {error}",
            exc_info=True,
        )
        return False

    still_missing = [path for path in required_files if not os.path.exists(path)]
    if still_missing:
        logger.error(
            f"[ERROR] [{domain}] Files still missing after processing: "
            f"{still_missing}"
        )
        return False
    logger.info(f"[TRAINING] [{domain}] Data processing pipeline completed")
    return True


def _check_and_prepare_joint_dataset(config):
    domains = config["domains"]
    if len(domains) != 2:
        logger.info("[TRAINING] Single dataset mode, skipping joint dataset creation")
        return True

    data_path = config.get("data_path", "../data/")
    joint_dataset_name = "+".join(domains)
    dataset_type = "only_overlap_users" if config["only_overlap_users"] else "all_users"
    split_dir = (
        f"WarmValid{config['warm_valid_ratio']}_"
        f"WarmTest{config['warm_test_ratio']}_"
        f"ColdValid{config['t_cold_valid']}_"
        f"ColdTest{config['t_cold_test']}_"
        f"{'shuffle' if config['shuffle_user_sequence'] else 'noshuffle'}"
    )
    if config["only_overlap_users"]:
        split_dir += f'_{config["k_cores"]}cores'
    joint_path = os.path.join(
        data_path,
        config["dataset"],
        joint_dataset_name,
        dataset_type,
        split_dir,
    )

    required_joint_files = [
        os.path.join(joint_path, "all_users.json"),
        os.path.join(joint_path, "id_mapping.json"),
        os.path.join(joint_path, "train_src.pkl"),
        os.path.join(joint_path, "train_tgt.pkl"),
        os.path.join(joint_path, "valid_cold_tgt.pkl"),
        os.path.join(joint_path, "test_cold_tgt.pkl"),
        os.path.join(joint_path, "valid_warm_tgt.pkl"),
        os.path.join(joint_path, "test_warm_tgt.pkl"),
    ]
    for modality in config["modalities"]:
        if modality.get("enabled", False):
            filename = f'{modality["name"]}_final_emb_{modality["emb_pca"]}.npy'
            required_joint_files.append(
                os.path.join(joint_path, "modality_emb", filename)
            )

    missing_files = [path for path in required_joint_files if not os.path.exists(path)]
    if not missing_files:
        logger.info(
            "[TRAINING] All required joint dataset files (including "
            "embeddings) already exist. Skipping creation."
        )
        return True

    logger.info(
        "[TRAINING] One or more joint dataset files are missing. "
        "Recreating all joint files."
    )
    logger.info(f"Missing files: {missing_files}")
    try:
        create_joint_dataset(domains=domains, config=config)
    except Exception as error:
        logger.error(
            f"[ERROR] Failed to create joint dataset: {error}", exc_info=True
        )
        return False

    still_missing = [path for path in required_joint_files if not os.path.exists(path)]
    if still_missing:
        logger.error(
            "[ERROR] Joint dataset creation failed, still missing: "
            f"{still_missing}"
        )
        return False
    return True


def check_and_prepare_Douban(config):
    domains = config["domains"]
    logger.info(
        "[TRAINING] Starting parallel data preparation for dataset: Douban, "
        f"processing domains individually: {domains}"
    )
    with ThreadPoolExecutor(max_workers=len(domains)) as executor:
        futures = {
            executor.submit(check_and_prepare_Douban_single, config, domain): domain
            for domain in domains
        }
        results = {}
        for future in as_completed(futures):
            domain = futures[future]
            try:
                results[domain] = future.result()
            except Exception:
                logger.error(
                    f"[ERROR] [{domain}] Exception during data preparation",
                    exc_info=True,
                )
                results[domain] = False

    failed_domains = [domain for domain, success in results.items() if not success]
    if failed_domains:
        logger.error(f"[ERROR] Failed to prepare data for domains: {failed_domains}")
        return False
    return _check_and_prepare_joint_dataset(config)


def check_and_prepare_dataset(config):
    dataset_handlers = {
        "Amazon2014": check_and_prepare_Amazon2014,
        "Douban": check_and_prepare_Douban,
    }
    handler = dataset_handlers.get(config["dataset"])
    if handler is None:
        raise ValueError(f'Unsupported dataset: {config["dataset"]}')
    return handler(config)
