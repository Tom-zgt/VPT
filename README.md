# VPT: Enhancing Video Physical Consistency via Role-aware Joint Training and Modality-decoupled Denoising



> **Guangting Zheng**, **Haojing Chen**, Hao Li, Jingtao Zhang, Zhen Yang, Xiaosong Jia, Xue Yang, Shaofeng Zhang, Yanyong Zhang
>
> 1USTC   2UESTC   3Fudan University   4Georgia Tech   5SJTU
>
> Equal contribution

**VPT** is a fine-tuning framework that improves the *physical consistency* of pretrained
video diffusion models (e.g. Wan2.1-T2V) while preserving their visual quality.

  
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

## Method


| Component                        | Description                                                                                                                               |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Role-aware joint training**    | Groups scene entities into agents / controlled objects / passive objects / background so different physical roles are modeled explicitly. |
| **Modality-decoupled denoising** | Visual and auxiliary (flow / seg) channels get *independent* noise levels, avoiding capacity conflicts and preserving the visual prior.   |
| **Loss-weight decay**            | The auxiliary-loss weight decays over training, making auxiliary signals soft constraints and mitigating recursive inference error.       |
| **Cross-step auto-guidance**     | Inference-time guidance over auxiliary streams that strengthens physical dynamics with no extra training.                                 |


## Results

### VideoPhy & VideoPhy-2

**Overall** Semantic Adherence (SA) and Physical Commonsense (PC). VPT is applied on top of both the 1.3B and 14B Wan2.1-T2V backbones.

| Model | VideoPhy SA | VideoPhy PC | VideoPhy-2 SA | VideoPhy-2 PC |
| --- | :---: | :---: | :---: | :---: |
| Wan2.1-T2V-1.3B | 47.7 | 21.2 | 19.3 | 53.7 |
| &nbsp;&nbsp;+ Full Fine-tune | 45.1 | 20.9 | 18.9 | 53.6 |
| &nbsp;&nbsp;+ VideoJAM | 49.1 | 22.1 | 20.6 | 54.0 |
| &nbsp;&nbsp;**+ VPT (Ours)** | **66.5** | **25.0** | **22.5** | **55.1** |
| Wan2.1-T2V-14B | 56.1 | 23.2 | 21.9 | 52.9 |
| &nbsp;&nbsp;+ Full Fine-tune | 62.1 | 21.5 | 20.7 | 54.0 |
| &nbsp;&nbsp;**+ VPT (Ours)** | **67.7** | **30.0** | **23.3** | **59.9** |

On the 1.3B backbone, VPT lifts VideoPhy Overall SA 47.7 → 66.5 (**+39.4% rel.**) and PC 21.2 → 25.0 (**+17.9% rel.**); on 14B, SA 56.1 → 67.7 and PC 23.2 → 30.0. Per-category (solid–solid / solid–fluid / fluid–fluid) breakdowns are in the paper.

### VBench

Wan2.1-T2V-1.3B backbone, official **raw** prompts (no prompt enhancement), 81 frames @ 480×832, 16 FPS.

| Dimension | Wan2.1-1.3B | + Full FT | + VideoJAM | + VPT (Ours) |
| --- | :---: | :---: | :---: | :---: |
| **Total Score** | 76.93 | 78.71 | 78.76 | **79.58** |
| Quality Score | 79.81 | 81.26 | 81.18 | **83.25** |
| Semantic Score | 65.43 | 68.47 | 69.08 | 64.86 |
| Subject Consistency | 91.83 | 93.59 | 91.51 | 92.65 |
| Background Consistency | 94.71 | 95.81 | 96.01 | **97.27** |
| Temporal Flickering | 99.17 | 99.36 | 99.13 | 98.64 |
| Motion Smoothness | 96.51 | 97.21 | 96.05 | **97.85** |
| Dynamic Degree | 65.00 | 54.08 | 73.88 | 70.83 |
| Object Class | 76.09 | 79.90 | 79.22 | 73.43 |
| Color | 89.93 | 88.57 | 88.92 | 86.99 |
| Human Action | 74.60 | 78.98 | 79.20 | 74.80 |
| Multiple Objects | 53.66 | 58.20 | 59.66 | 51.92 |
| Scene | 20.03 | 28.55 | 28.34 | 20.13 |
| Spatial Relationship | 62.37 | 63.31 | 66.17 | **73.35** |

VPT achieves the best overall and quality scores, with clear gains in background consistency, motion smoothness, and spatial relationship — i.e. more stable dynamics and better object-relation modeling. See the paper for full tables and ablations.

## Repository layout

```
VPT/
├── README.md                     # this file
├── LICENSE
├── requirements.txt
├── setup.py                      # optional: pip install -e .
├── train.py                      # entry point (inference = --train_steps 1, runs validation only)
├── finetrainers/                 # runtime package (model, pipeline, trainer, data, parallel)
│   └── models/wan/
│       ├── base_specification.py # WanModelSpecification + validation()/video save
│       └── custome.py            # CustomWanPipeline (denoise loop, CFG modes)
├── examples/training/sft/wan/3dgs_dissolve/
│   ├── infer.sh                  # launcher (standard torchrun, single- & multi-node)
│   ├── training.json             # required by --dataset_config (see caveat)
│   ├── videophy.json videophy2.json ood.json vbench*.json
│   └── build_eval_prompts_from_videorepa.py
├── assets/teaser.gif
└── docs/                         # project page (index.html + demo videos)
```

## How inference works

There is no separate `infer.py`. Inference reuses `train.py` with `--train_steps 1`, which
**skips the training loop and runs validation only**: for every prompt in the validation
JSON it generates one `.mp4` into `--save_videos_dir`.

```
train.py → finetrainers SFT trainer → _validate()
        → WanModelSpecification.validation()
        → CustomWanPipeline.__call__()  (denoise + inner / text CFG)
        → imageio writes the video
```

Role/flow/seg (triple-discrete) timestep conditioning is enabled **automatically**:
`CustomWanPipeline` detects `cond_proj` weights in the LoRA checkpoint and swaps in
`WanTripleDiscreteTimeTextImageEmbedding`. No `multi_timestep_`* flag is needed at inference.

> **Caveat:** even with `--train_steps 1`, the trainer still initializes the training
> dataloader (`--dataset_config training.json`) and consumes one batch to load the
> tokenizer / text-encoder / VAE before validation. Keep `training.json` and point its
> `data_root` at a small readable video set.

## Installation

```bash
git clone https://github.com/Tom-zgt/VPT.git
cd VPT
pip install -r requirements.txt
pip install -U "transformers>=4.49.0,<5.0.0" "accelerate>=0.34.0" \
               "diffusers==0.36.0" "tokenizers>=0.20.0" "peft>=0.13.0,<0.19"
# optional: pip install -e .
```

You need two things:

- **Base model**: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (a HuggingFace repo id or a local dir).
- **VPT LoRA checkpoint**: the trained `pytorch_lora_weights.safetensors`.

## Usage

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
`NODE_RANK`. Example — 4 nodes × 8 GPUs:

```bash
# on every node (change NODE_RANK to 0,1,2,3):
NNODES=4 NODE_RANK=0 NPROC_PER_NODE=8 \
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
BENCHMARK=videophy \
LORA_WEIGHTS_PATH=/path/to/pytorch_lora_weights.safetensors \
SAVE_VIDEOS_DIR=./outputs/videophy \
bash examples/training/sft/wan/3dgs_dissolve/infer.sh
```

Internally `infer.sh` launches:

```bash
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} \
         --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
         train.py ...
```

Topology env vars (all optional; single-node defaults shown):
`NNODES=1`, `NODE_RANK=0`, `NPROC_PER_NODE=<#visible GPUs>`, `MASTER_ADDR=127.0.0.1`,
`MASTER_PORT=29500`. `--dp_degree` must equal `NNODES × NPROC_PER_NODE` (infer.sh computes this).

### Guidance modes

- `WAN_VALIDATION_INNER_GUIDANCE=1` → chained (inner) CFG over text + flow + seg, controlled by
`WAN_VALIDATION_{GUIDANCE,FLOW_GUIDANCE,SEG_GUIDANCE}_SCALE`.
- `WAN_VALIDATION_INNER_GUIDANCE=0` → standard text-only CFG (`--no-validation_enable_inner_guidance`).

### Benchmarks

`BENCHMARK` selects the validation prompt file: `videophy | videophy2 | vbench | vbench2 | ood`.
This repo only **generates** videos; score them with the separate benchmark tools
(VideoPhy / VideoPhy2 / VBench), which are intentionally not bundled here.

## Project page

The `docs/` folder is a self-contained project page (`docs/index.html`) with all qualitative
comparisons and reconstruction demos. Open it locally, or enable **GitHub Pages** on the
`docs/` folder to publish it.

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
VideoJAM, VideoPhy, and VBench for their benchmarks and baselines.

## License

Released under the [Apache 2.0](LICENSE) license.