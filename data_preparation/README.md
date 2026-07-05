# Data Preparation

VPT fine-tunes Wan2.1-T2V on the **[WISA-80K](https://huggingface.co/datasets/qihoo360/WISA-80K)**
dataset augmented with two auxiliary modalities:

1. **Optical flow** — extracted with [RAFT](https://arxiv.org/abs/2003.12039) (`raft_flow/`).
2. **Role / semantic maps** — produced by Qwen3-VL (role assignment) + SAM3 (masks) (`role_map/`).

Both auxiliary streams are encoded by the **same Wan video VAE** as the RGB video, so the
final training input is a 9-channel joint latent `[video ⊕ flow ⊕ role]`.

```
data_preparation/
├── raft_flow/            # RGB video -> optical-flow (VAE-encoded)
│   ├── core/             # RAFT model (Teed & Deng, ECCV 2020)
│   ├── latent.py         # flow -> Wan-VAE latents (.pt)
│   ├── latent_videos.py  # flow -> decoded flow .mp4 (for inspection / seg_feature_root)
│   └── download_models.sh
└── role_map/             # RGB video -> role/semantic map (grayscale .mp4)
    ├── src/              # qwen_stage, sam3_stage, mask_ops, pipeline, rgb / rgb2
    └── prompts/qwen_system.txt
```

Suggested layout (matches the training defaults in `examples/.../training.json` and
`base_specification.py`):

```
data/
├── videos_data/videos/        # source RGB clips from WISA-80K
└── latent_data/
    ├── flow_videos/           # RAFT optical-flow outputs
    └── seg_videos/            # role/semantic-map outputs
```

---

## 0. Download WISA-80K

```bash
# clips live under data/videos of the dataset repo (~454 GB total, sharded *.zip)
huggingface-cli download qihoo360/WISA-80K --repo-type dataset \
  --revision dddbd5683581c2ebf0b463e2b1c3342b2094bfb3 \
  --include "data/videos/*" --local-dir ./WISA-80K

# unzip the shards into data/videos_data/videos/
mkdir -p data/videos_data/videos
for z in ./WISA-80K/data/videos/*.zip; do unzip -n "$z" -d data/videos_data/videos/; done
```

Source: <https://huggingface.co/datasets/qihoo360/WISA-80K/tree/dddbd5683581c2ebf0b463e2b1c3342b2094bfb3/data/videos>

## 1. Optical flow (RAFT)

```bash
cd data_preparation/raft_flow
./download_models.sh            # fetches raft-things.pth into models/

# flow -> Wan-VAE latents (.pt)
python latent.py \
  --video_folder ../../data/videos_data/videos \
  --output_dir   ../../data/latent_data/flow_videos \
  --raft_model   models/raft-things.pth \
  --vae_checkpoint Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --num_gpus 8

# (optional) decode flow latents back to .mp4 for visual inspection
python latent_videos.py --video_folder ../../data/videos_data/videos \
  --output_dir ../../data/latent_data/flow_videos --num_gpus 8
```

## 2. Role / semantic maps (Qwen3-VL + SAM3)

Assigns every entity a physical role — **agent (255)**, **agent-controlled object (170)**,
**passive object (85)**, **background (0)** — and rasterizes it into a single-channel map.

Setup (see comments in `role_map/src`): Python 3.12+, CUDA 12.6+, `transformers`,
plus SAM3 from source:

```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate einops tiktoken opencv-python huggingface_hub
git clone https://github.com/facebookresearch/sam3.git && (cd sam3 && pip install -e .)
```

Run (shard across GPUs with `--start_idx` / `--process_len`):

```bash
cd data_preparation/role_map
CUDA_VISIBLE_DEVICES=0 python -m src.rgb2 \
  --video ../../data/videos_data/videos \
  --system-prompt prompts/qwen_system.txt \
  --output-dir ../../data/latent_data/seg_videos \
  --model-id Qwen/Qwen3-VL-30B-A3B-Instruct \
  --max-new-tokens 512 --start_idx 0 --process_len 10000
```

Stages:
1. **Qwen3-VL** labels entity categories and roles → `metadata/<video>_categories.json`.
2. **SAM3** segments each category (text prompt on frame 0, propagated across the clip).
3. Roles are merged into a grayscale semantic channel and saved as `<video>_S.mp4`.

> Notes
> - `Qwen/Qwen3-VL-30B-A3B-Instruct` and `facebook/sam3` are gated on Hugging Face; request
>   access and `huggingface-cli login` first. Never commit access tokens.
> - `raft_flow/` retains RAFT's original BSD license (`raft_flow/LICENSE`).
