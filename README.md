# VPT: Enhancing Video Physical Consistency via Role-aware Joint Training and Modality-decoupled Denoising

[Code](https://github.com/Tom-zgt/VPT)
[Paper](https://github.com/Tom-zgt/VPT)
[Project Page](docs/index.html)
License

> **Guangting Zheng**, **Haojing Chen**, Hao Li, Jingtao Zhang, Zhen Yang, Xiaosong Jia, Xue Yang, Shaofeng Zhang, Yanyong Zhang
>
> 1USTC   2UESTC   3Fudan University   4Georgia Tech   5SJTU   (equal contribution)

**VPT** is a fine-tuning framework that improves the *physical consistency* of pretrained
video diffusion models (e.g. Wan2.1-T2V) while preserving their visual quality.

VPT vs. Wan2.1-1.3B vs. VideoJAM

*"A wine bottle pours a red blend into a glass." VPT produces more physically consistent motion and interactions.*

> More qualitative comparisons (Wan2.1-1.3B & 14B) and reconstruction demos are on the
> **[project page](docs/index.html)** — open `docs/index.html` in a browser, or host `docs/` with GitHub Pages.

## Abstract

While modern video diffusion models excel in visual fidelity, maintaining long-range
physical consistency remains a formidable challenge. Conventional pixel-reconstruction
objectives mainly focus on appearance details and often fail to capture the underlying
dynamics of a scene. To mitigate this, recent efforts integrate auxiliary modalities
(e.g. optical flow) to introduce physics priors via joint training with video appearance.
However, these methods have three main limitations: (1) they do not distinguish the
different motion patterns of different entity types; (2) joint modeling of visual and
auxiliary modalities can cause capacity conflicts and weaken the pretrained visual prior;
and (3) auxiliary modalities may accumulate errors during inference. To address these
issues, we propose **VPT**. VPT introduces a **role-aware** signal that groups entities into
*agents, controlled objects, passive objects, and background*; a **modality-decoupled
denoising** strategy that assigns independent noise levels to visual and auxiliary
channels; a **loss-weight decay** strategy that turns auxiliary modalities into soft
constraints; and **cross-step auto-guidance** to further strengthen physical dynamics.
VPT achieves relative gains of **39.4% in VideoPhy SA** and **17.9% in VideoPhy PC** over
Wan2.1-T2V-1.3B, with consistent improvements on VideoPhy-2.
## Methods

<p align="center">
  <img src="./assets/VPT_motion_tune_method.jpg" alt="图片描述" width="100%">
</p>

## Results

### VideoPhy & VideoPhy-2

**Overall** Semantic Adherence (SA) and Physical Commonsense (PC). VPT is applied on top of both the 1.3B and 14B Wan2.1-T2V backbones.


| Model            | VideoPhy SA | VideoPhy PC | VideoPhy-2 SA | VideoPhy-2 PC |
| ---------------- | ----------- | ----------- | ------------- | ------------- |
| Wan2.1-T2V-1.3B  | 47.7        | 21.2        | 19.3          | 53.7          |
| + Full Fine-tune | 45.1        | 20.9        | 18.9          | 53.6          |
| + VideoJAM       | 49.1        | 22.1        | 20.6          | 54.0          |
| **+ VPT (Ours)** | **66.5**    | **25.0**    | **22.5**      | **55.1**      |
| Wan2.1-T2V-14B   | 56.1        | 23.2        | 21.9          | 52.9          |
| + Full Fine-tune | 62.1        | 21.5        | 20.7          | 54.0          |
| **+ VPT (Ours)** | **67.7**    | **30.0**    | **23.3**      | **59.9**      |


### VBench

Wan2.1-T2V-1.3B backbone, official **raw** prompts (no prompt enhancement), 81 frames @ 480×832, 16 FPS.


| Metric          | Wan2.1-1.3B | + Full FT | + VideoJAM | + VPT (Ours) |
| --------------- | ----------- | --------- | ---------- | ------------ |
| **Total Score** | 76.93       | 78.71     | 78.76      | **79.58**    |
| Quality Score   | 79.81       | 81.26     | 81.18      | **83.25**    |
| Semantic Score  | 65.43       | 68.47     | 69.08      | 64.86        |


VPT achieves the best total and quality scores. See the paper for the full per-dimension breakdown and ablations.

## Installation

```bash
git clone https://github.com/Tom-zgt/VPT.git
cd VPT
pip install -r requirements.txt
pip install -U "transformers>=4.49.0,<5.0.0" "accelerate>=0.34.0" \
               "diffusers==0.36.0" "tokenizers>=0.20.0" "peft>=0.13.0,<0.19"
# optional: pip install -e .
```

## Pretrained checkpoints

To run inference you need the base Wan model plus our trained VPT LoRA:

- **Base model**: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (HuggingFace repo id or a local dir).
- **VPT LoRA (ours)**: the trained `pytorch_lora_weights.safetensors`.


| Model    | Backbone        | Download             |
| -------- | --------------- | -------------------- |
| VPT LoRA | Wan2.1-T2V-1.3B | **[Link](https://huggingface.co/zhengzhou/VPT/blob/main/pytorch_lora_weights.safetensors)** |



Download the LoRA and pass its path to `infer.sh` / `train.sh` via `LORA_WEIGHTS_PATH`.

## Dataset & data preparation

VPT is fine-tuned on the **[WISA-80K](https://huggingface.co/datasets/qihoo360/WISA-80K)**
dataset. Download the clips from:

[https://huggingface.co/datasets/qihoo360/WISA-80K/tree/dddbd5683581c2ebf0b463e2b1c3342b2094bfb3/data/videos](https://huggingface.co/datasets/qihoo360/WISA-80K/tree/dddbd5683581c2ebf0b463e2b1c3342b2094bfb3/data/videos)

```bash
huggingface-cli download qihoo360/WISA-80K --repo-type dataset \
  --revision dddbd5683581c2ebf0b463e2b1c3342b2094bfb3 \
  --include "data/videos/*" --local-dir ./WISA-80K
mkdir -p data/videos_data/videos
for z in ./WISA-80K/data/videos/*.zip; do unzip -n "$z" -d data/videos_data/videos/; done
```

Each clip is augmented with two auxiliary modalities before training:

- **Optical flow** via RAFT — `data_preparation/raft_flow/`
- **Role / semantic maps** via Qwen3-VL + SAM3 — `data_preparation/role_map/`

Both are encoded by the same Wan video VAE, forming the 9-channel joint latent
`[video ⊕ flow ⊕ role]`. Full step-by-step instructions are in
[data_preparation/README.md](data_preparation/README.md).

Point `examples/training/sft/wan/3dgs_dissolve/training.json` (`data_root`) at your RGB
clips; the flow / role latents are read from `data/latent_data/{flow_videos,seg_videos}`
(the `WanModelSpecification` defaults).

## Training

Fine-tune the VPT LoRA with `train.sh`. Role/flow (triple-discrete) timestep decoupling
and auxiliary-loss annealing are enabled automatically once the flow/role latents exist.

### Single node (all visible GPUs)

```bash
OUTPUT_DIR=./outputs/vpt_train \
PRETRAINED_MODEL_PATH=Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
bash examples/training/sft/wan/3dgs_dissolve/train.sh
```

Defaults follow the paper's 1.3B setup: LoRA rank/alpha 32, per-GPU `batch_size 6`
(global batch 48 on 8 GPUs), AdamW, `train_steps 2000` (the released LoRA is the
step-2000 checkpoint), checkpoint every 1000 steps. Override via env vars, e.g.
`TRAIN_STEPS=2000 BATCH_SIZE=6 LR=5e-5 RESUME=latest REPORT_TO=wandb`.

### Multi-node (standard `torchrun`)

Run on **every** node with the same `MASTER_ADDR` / `MASTER_PORT` and a distinct
`NODE_RANK`. The 14B model in the paper uses 32 GPUs (4 nodes × 8, `batch_size 1`):

```bash
# on every node (change NODE_RANK to 0,1,2,3):
NNODES=4 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
OUTPUT_DIR=./outputs/vpt_train \
bash examples/training/sft/wan/3dgs_dissolve/train.sh
```

Trained LoRA weights land in `${OUTPUT_DIR}/.../<step>/pytorch_lora_weights.safetensors`;
pass that to `infer.sh` via `LORA_WEIGHTS_PATH`.

## Inference

### Single node (all visible GPUs)

```bash
BENCHMARK=videophy \
PRETRAINED_MODEL_PATH=Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
LORA_WEIGHTS_PATH=/path/to/pytorch_lora_weights.safetensors \
SAVE_VIDEOS_DIR=./outputs/videophy \
WAN_VALIDATION_GUIDANCE_SCALE=1.8 \
WAN_VALIDATION_INNER_GUIDANCE=1 \
bash examples/training/sft/wan/3dgs_dissolve/infer.sh
```

### Multi-node (standard `torchrun`)

Run `infer.sh` on **every** node with the same `MASTER_ADDR` / `MASTER_PORT` and a distinct
`NODE_RANK`. Example — 2 nodes × 8 GPUs:

```bash
# on every node (change NODE_RANK to 0,1):
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
BENCHMARK=videophy \
LORA_WEIGHTS_PATH=/path/to/pytorch_lora_weights.safetensors \
SAVE_VIDEOS_DIR=./outputs/videophy \
bash examples/training/sft/wan/3dgs_dissolve/infer.sh
```

Topology env vars (all optional; single-node defaults shown):
`NNODES=1`, `NODE_RANK=0`, `NPROC_PER_NODE=<#visible GPUs>`, `MASTER_ADDR=127.0.0.1`,
`MASTER_PORT=29500`. `--dp_degree` must equal `NNODES × NPROC_PER_NODE` (infer.sh computes this).

### Guidance modes

- `WAN_VALIDATION_INNER_GUIDANCE=1` → chained (inner) CFG over text + flow + seg, controlled by
`WAN_VALIDATION_{GUIDANCE,FLOW_GUIDANCE,SEG_GUIDANCE}_SCALE`. Setting
`WAN_VALIDATION_FLOW_GUIDANCE_SCALE=0` and `WAN_VALIDATION_SEG_GUIDANCE_SCALE=0` keeps only
the text CFG term inside the chained formulation (the setting used for the paper's VideoPhy numbers).


### Benchmarks

`BENCHMARK` selects the validation prompt file: `videophy | videophy2 | vbench`.
This repo only **generates** videos; score them with the separate benchmark tools
(VideoPhy / VideoPhy2 / VBench), which are intentionally not bundled here.

## Reproducing the paper results

The **VideoPhy** and **VideoPhy-2** numbers in the paper are produced with **guidance scale
2.2** (inner CFG, flow/seg scales = 0) and evaluated on **16 GPUs (2 nodes × 8)**.

```bash
# run on BOTH nodes (change NODE_RANK to 0 and 1), once for BENCHMARK=videophy and once for =videophy2:
NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
BENCHMARK=videophy \
LORA_WEIGHTS_PATH=/path/to/vpt_lora/pytorch_lora_weights.safetensors \
SAVE_VIDEOS_DIR=./outputs/videophy_gs2.2 \
WAN_VALIDATION_GUIDANCE_SCALE=2.2 \
WAN_VALIDATION_FLOW_GUIDANCE_SCALE=0 \
WAN_VALIDATION_SEG_GUIDANCE_SCALE=0 \
WAN_VALIDATION_INNER_GUIDANCE=1 \
bash examples/training/sft/wan/3dgs_dissolve/infer.sh
```

Then score the generated videos with the official
[VideoPhy](https://github.com/Hritikbansal/videophy) / VideoPhy-2 evaluators.

> ⚠️ **Important — VideoPhy / VideoPhy-2 are noisy benchmarks.** The SA/PC scores are
> sensitive to the sampling setup, so numbers are **not** perfectly deterministic:
>
> - **GPU count matters.** Because generation is sharded per rank (per-rank seeding), a
> **16-GPU** run and an **8-GPU** run produce different videos for the same prompts, and in our experience **16-GPU evaluation scores higher than 8-GPU**. 
> - **Seed matters.** Different `--seed` values also shift the scores.
>
> For exact reproducibility we release the **videos we generated** from the two benchmarks'
> prompts (used to compute the reported numbers):
>
> - VideoPhy generated videos: **[Link](https://huggingface.co/zhengzhou/VPT/blob/main/videophy_videos.zip)**
> - VideoPhy-2 generated videos: **[Link](https://huggingface.co/zhengzhou/VPT/blob/main/videophy2_videos.zip)**

## Project page

The `docs/` folder is a self-contained project page (`docs/index.html`) with all qualitative
comparisons and reconstruction demos. 

## Citation

```bibtex
@article{zheng2026vpt,
  title   = {Enhancing Video Physical Consistency via Role-aware Joint Training and Modality-decoupled Denoising},
  author  = {Zheng, Guangting and Chen, Haojing and Li, Hao and Zhang, Jingtao and
             Yang, Zhen and Jia, Xiaosong and Yang, Xue and Zhang, Shaofeng and Zhang, Yanyong},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

Built on [Wan2.1](https://github.com/Wan-Video/Wan2.1) and the
[finetrainers](https://github.com/a-r-r-o-w/finetrainers) framework. We thank the authors of
VideoJAM, VideoREPA, VideoPhy, and VBench for their benchmarks and baselines.

## License

Released under the [Apache 2.0](LICENSE) license.