# VPT: Enhancing Video Physical Consistency via Role-aware Joint Training and Modality-decoupled Denoising

<p align="center">
  <a href="https://github.com/Tom-zgt/VPT"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-black?logo=github"></a>
  <a href="https://github.com/Tom-zgt/VPT"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv"></a>
  <a href="docs/index.html"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2563eb"></a>
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green">
</p>

> **Guangting Zheng**\*, **Haojing Chen**\*, Hao Li, Jingtao Zhang, Zhen Yang, Xiaosong Jia, Xue Yang, Shaofeng Zhang, Yanyong Zhang
> <br/><sup>1</sup>USTC &nbsp; <sup>2</sup>UESTC &nbsp; <sup>3</sup>Fudan University &nbsp; <sup>4</sup>Georgia Tech &nbsp; <sup>5</sup>SJTU
> <br/>\*Equal contribution

**VPT** is a fine-tuning framework that improves the *physical consistency* of pretrained
video diffusion models (e.g. Wan2.1-T2V) while preserving their visual quality.

<p align="center">
  <img src="assets/teaser.gif" width="100%" alt="VPT vs. Wan2.1-1.3B vs. VideoJAM — 'A wine bottle pours a red blend into a glass'">
  <br/>
  <em>"A wine bottle pours a red blend into a glass." VPT produces more physically consistent motion and interactions.</em>
</p>

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

| Component | Description |
| --- | --- |
| **Role-aware joint training** | Groups scene entities into agents / controlled objects / passive objects / background so different physical roles are modeled explicitly. |
| **Modality-decoupled denoising** | Visual and auxiliary (flow / seg) channels get *independent* noise levels, avoiding capacity conflicts and preserving the visual prior. |
| **Loss-weight decay** | The auxiliary-loss weight decays over training, making auxiliary signals soft constraints and mitigating recursive inference error. |
| **Cross-step auto-guidance** | Inference-time guidance over auxiliary streams that strengthens physical dynamics with no extra training. |

## Results

VideoPhy — SA (Semantic Adherence) and PC (Physical Commonsense), VPT applied on Wan2.1-T2V-1.3B:

| Model | VideoPhy SA | VideoPhy PC |
| --- | :---: | :---: |
| Wan2.1-T2V-1.3B (base) | — | — |
| **+ VPT (Ours)** | **+39.4%** (rel.) | **+17.9%** (rel.) |

VPT also improves VideoPhy-2. See the paper for full tables and ablations.

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
`WanTripleDiscreteTimeTextImageEmbedding`. No `multi_timestep_*` flag is needed at inference.

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
