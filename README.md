# LSDCDR

This source package contains the code required to train and evaluate LSDCDR on the Amazon2014 Clothing, Shoes and Jewelry to Sports and Outdoors cold-start recommendation task. Dataset configuration files for both Amazon2014 and Douban are included.

## Package layout

```text
LSDCDR/
|-- data/                  # Created automatically on the first run
|-- src/
|   |-- common/            # Training and evaluation framework
|   |-- configs/           # Dataset and fixed best-model configuration
|   |-- log/               # Runtime logs
|   |-- models/lsdcdr.py   # LSDCDR model and semantic preprocessing
|   `-- utils/             # Data preparation, loading, and metrics
`-- README.md
```

Only the LSDCDR model and its Cloth-to-Sport configuration are included. The released configuration matches the setting used for the reported result: Recall@20 = 0.0763 and NDCG@20 = 0.0700.

## Environment

The experiments were run with Python 3.8, PyTorch 1.12.0, and CUDA 11.6. A tested dependency set is:

```bash
pip install \
  numpy==1.24.4 pandas==2.0.3 scipy==1.10.1 pyyaml==6.0.2 \
  tqdm==4.66.5 requests==2.32.3 openai==1.52.1 \
  scikit-learn==1.3.2 matplotlib==3.7.2 tensorboard==2.12.1 \
  kaggle==1.6.17
```

Install the PyTorch build appropriate for the local CUDA environment separately.

## Data and semantic embeddings

The `data/` directory is intentionally empty. On the first run, the benchmark downloads the selected Amazon2014 or Douban domains, creates the cold-start split, and caches all processed files under `data/`. Amazon2014 data is obtained from the Stanford SNAP repository. Douban data is obtained from Kaggle and requires `kaggle_user_name` and `kaggle_api_token` in `src/configs/dataset/Douban.yaml` or the equivalent Kaggle environment configuration.

Semantic preprocessing uses `gpt-4o-mini` for preference-tag extraction and `text-embedding-3-large` for text embeddings. The semantic-embedding settings are defined in `src/configs/model/LSDCDR.yaml`. Credential fields are retained in both dataset configuration files with empty values. They can be filled locally or supplied through environment variables; credentials are not stored in this package.

```bash
export OPENAI_API_KEY="your-api-key"
# Optional for an OpenAI-compatible endpoint:
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

The first run performs API-backed semantic preprocessing and can take substantially longer than later runs. Generated metadata and embeddings are cached and reused automatically. Only training interactions are used to construct semantic profiles.

## Evaluation

Run the command from `LSDCDR/src`:

```bash
cd LSDCDR/src
CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --model LSDCDR \
  --dataset Amazon2014 \
  --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors
```

The fixed configuration uses seed 999, 200 diffusion steps, a cosine noise schedule, pseudo Top-K of 5, and pseudo-supervision starting at epoch 80. Model selection uses validation Recall@20, and cold-start test metrics are written to `src/log/`.
