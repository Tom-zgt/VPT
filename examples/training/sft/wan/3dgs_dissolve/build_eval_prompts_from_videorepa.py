#!/usr/bin/env python3
"""Build Wan validation JSON aligned with VideoREPA inference/generate.py.

VideoPhy / VideoPhy2: same sources as VideoREPA-main/inference (videophy.txt +
videophy_detailed.txt, videophy2.csv). Generation prompt = long / upsampled,
ori_caption = short; filenames follow generate.py (short caption joined by '_').

VBench: prompts from official ``VBench_full_info.json`` (same ``prompt_en`` + dedupe
as ``convert2.py``). Resolution / frame count default to **480×832, 81 frames** (same
as Wan ``CustomWanPipeline`` / VideoPhy eval); filenames for eval use ``prompt_en``
(see ``WAN_EVAL_STYLE`` handling in Wan validation).

VBench-1.0 sample counts follow ``VBench-master/prompts/README.md``: **5** videos per
``prompt_en`` by default, **25** when the prompt appears under dimension
``temporal_flickering`` (static-filter coverage). Use CLI ``--vbench-plain-prompt`` (or
``VBENCH_PLAIN_PROMPT=1`` in ``infer.sh``) to disable GPT aug and set ``caption=prompt_en``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def _wan_eval_item(
    long_prompt: str,
    short_prompt: str,
    *,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    val_num_per_prompt: Optional[int] = None,
) -> dict:
    out = {
        "caption": long_prompt,
        "ori_caption": short_prompt,
        "image_path": None,
        "video_path": None,
        "num_inference_steps": num_inference_steps,
        "height": height,
        "width": width,
        "num_frames": num_frames,
    }
    if val_num_per_prompt is not None:
        out["val_num_per_prompt"] = int(val_num_per_prompt)
    return out


def _wan_eval_item_single_prompt(
    prompt: str,
    *,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
) -> dict:
    """VBench-style: ``ori_caption`` = benchmark ``prompt_en`` (used for eval filenames)."""
    p = prompt.strip()
    return _wan_eval_item(
        p,
        p,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    )


def build_videophy(
    videorepa_root: Path,
    out_path: Path,
    *,
    short_name: str = "videophy.txt",
    detailed_name: str = "videophy_detailed.txt",
    label: str = "VideoPhy",
) -> None:
    txt = videorepa_root / "inference" / short_name
    detailed = videorepa_root / "inference" / detailed_name
    shorts = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    longs = [ln.strip() for ln in detailed.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(shorts) != len(longs):
        raise SystemExit(f"Line count mismatch: {txt} ({len(shorts)}) vs {detailed} ({len(longs)})")
    data = {"data": [_wan_eval_item(l, s) for s, l in zip(shorts, longs)]}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(data['data'])} {label} samples -> {out_path}")


def build_videophy2(videorepa_root: Path, out_path: Path) -> None:
    csv_path = videorepa_root / "inference" / "videophy2.csv"
    # Same layout as VideoREPA inference/generate.py (caption + upsampled_caption).
    df = pd.read_csv(csv_path)
    if "caption" not in df.columns or "upsampled_caption" not in df.columns:
        raise SystemExit(f"Expected columns caption, upsampled_caption in {csv_path}")
    rows = []
    for _, row in df.iterrows():
        short = str(row["caption"]).strip()
        long = str(row["upsampled_caption"]).strip()
        rows.append(_wan_eval_item(long, short))
    data = {"data": rows}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(data['data'])} VideoPhy2 samples -> {out_path}")


def _dedup_prompt_en_rows(original_data: list) -> list[str]:
    """Match NAE/convert2.py: prompt_en only, skip duplicates (including quoted form check)."""
    captions: list[str] = []
    for item in original_data:
        if "prompt_en" not in item:
            continue
        pe = item["prompt_en"]
        if not isinstance(pe, str):
            continue
        pe = pe.strip()
        if pe in captions or f'"{pe}"' in captions:
            continue
        captions.append(pe)
    return captions


def _vbench1_prompt_to_max_samples(original_data: list) -> dict[str, int]:
    """
    VBench-1.0 ``prompts/README.md``: 5 samples per prompt, except **25** for prompts
    used by the ``temporal_flickering`` dimension (after static filter).
    """
    out: dict[str, int] = {}
    for item in original_data:
        pe = item.get("prompt_en")
        if not isinstance(pe, str) or not pe.strip():
            continue
        pe = pe.strip()
        dims = item.get("dimension") or []
        n = 25 if "temporal_flickering" in dims else 5
        out[pe] = max(out.get(pe, 5), n)
    return out


def _vbench1_prompt_en_to_augmented(vbench_master: Path) -> Optional[dict[str, str]]:
    """
    Pair official short prompts with GPT-long captions the same way VBench-master does:
    ``all_dimension.txt`` and ``all_dimension_longer.txt`` are aligned **by line index**
    (see ``prompts/augmented_prompts/gpt_enhanced_prompts/README.md``).

    Do **not** align aug lines to ``_dedup_prompt_en_rows`` by index: JSON first-seen order
    diverges from ``all_dimension.txt`` order around ~index 746, which would pair wrong
    captions to ``ori_caption`` / eval filenames.
    """
    short_path = vbench_master / "prompts" / "all_dimension.txt"
    aug_path = (
        vbench_master
        / "prompts"
        / "augmented_prompts"
        / "gpt_enhanced_prompts"
        / "all_dimension_longer.txt"
    )
    if not short_path.is_file() or not aug_path.is_file():
        return None
    shorts = _load_lines(short_path)
    augs = _load_lines(aug_path)
    if len(shorts) != len(augs):
        raise SystemExit(
            f"VBench-1.0 aug line mismatch: {short_path} ({len(shorts)}) vs {aug_path} ({len(augs)})"
        )
    # First occurrence wins (``all_dimension.txt`` has 2 duplicate ``prompt_en`` lines with
    # different longer variants for one of them).
    out: dict[str, str] = {}
    for s, a in zip(shorts, augs):
        if s not in out:
            out[s] = a
    return out


def build_vbench_from_full_info(
    json_path: Path,
    out_path: Path,
    *,
    height: int,
    width: int,
    num_frames: int,
    aug_prompt_map: Optional[dict[str, str]] = None,
    val_num_per_prompt: Optional[int] = None,
    vbench1_prompt_samples: Optional[dict[str, int]] = None,
) -> None:
    original_data = json.loads(json_path.read_text(encoding="utf-8"))
    deduped = _dedup_prompt_en_rows(original_data)
    transformed: list[dict] = []
    for pe in deduped:
        if aug_prompt_map is not None:
            aug = aug_prompt_map.get(pe)
            long_p = aug.strip() if aug and aug.strip() else pe
        else:
            long_p = pe
        if vbench1_prompt_samples is not None:
            vn = vbench1_prompt_samples.get(pe, 5)
        else:
            vn = val_num_per_prompt
        transformed.append(
            _wan_eval_item(
                long_p,
                pe,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=50,
                val_num_per_prompt=vn,
            )
        )
    data = {"data": transformed}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(data['data'])} VBench prompts (deduped prompt_en) -> {out_path}")


def _load_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    here = Path(__file__).resolve().parent
    # .../NAE/videojam-master/examples/training/sft/wan/3dgs_dissolve -> parents[5] == NAE
    nae_root = here.parents[5]
    default_vr = nae_root / "VideoREPA-main"

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videorepa-root",
        type=Path,
        default=default_vr,
        help="VideoREPA-main (needs inference/videophy.txt, videophy_detailed.txt, videophy2.csv)",
    )
    ap.add_argument(
        "--nae-root",
        type=Path,
        default=nae_root,
        help="Repo root containing VBench_full_info.json",
    )
    ap.add_argument(
        "--vbench-master",
        type=Path,
        default=None,
        help="VBench-master checkout (GPT aug prompts). Default: <nae-root>/VBench-master",
    )
    ap.add_argument(
        "--vbench-plain-prompt",
        action="store_true",
        help=(
            "VBench-1.0: use official prompt_en for caption (no GPT aug from all_dimension_longer.txt). "
            "Filenames / ori_caption unchanged; geometry still 480x832x81."
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=here,
        help="Directory for generated *.json",
    )
    ap.add_argument(
        "which",
        choices=("videophy", "videophy2", "vbench", "all"),
        help="Which benchmark JSON to build (all = every benchmark below that has sources)",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vr = args.videorepa_root
    nae = args.nae_root
    vbench_master = args.vbench_master or (nae / "VBench-master")

    def run_videophy() -> None:
        if not (vr / "inference" / "videophy.txt").is_file():
            print(f"WARN: skip videophy (missing {vr}/inference/videophy.txt)", file=sys.stderr)
            return
        build_videophy(vr, out_dir / "videophy.json")

    def run_videophy2() -> None:
        if not (vr / "inference" / "videophy2.csv").is_file():
            print(f"WARN: skip videophy2 (missing {vr}/inference/videophy2.csv)", file=sys.stderr)
            return
        build_videophy2(vr, out_dir / "videophy2.json")

    def run_vbench() -> None:
        jp_vm = vbench_master / "vbench" / "VBench_full_info.json"
        jp = jp_vm if jp_vm.is_file() else (nae / "VBench_full_info.json")
        if not jp.is_file():
            print(f"WARN: skip vbench (missing {jp})", file=sys.stderr)
            return
        aug_map = None if args.vbench_plain_prompt else _vbench1_prompt_en_to_augmented(vbench_master)
        original_data = json.loads(jp.read_text(encoding="utf-8"))
        vbench1_counts = _vbench1_prompt_to_max_samples(original_data)
        # Geometry: Wan T2V default bucket (same as VideoPhy / CustomWanPipeline); counts: prompts/README.md (5 / 25).
        build_vbench_from_full_info(
            jp,
            out_dir / "vbench.json",
            height=480,
            width=832,
            num_frames=81,
            aug_prompt_map=aug_map,
            vbench1_prompt_samples=vbench1_counts,
        )

    if args.which == "all":
        run_videophy()
        run_videophy2()
        run_vbench()
    elif args.which == "videophy":
        if not vr.is_dir():
            raise SystemExit(f"VideoREPA root not found: {vr}")
        run_videophy()
    elif args.which == "videophy2":
        if not vr.is_dir():
            raise SystemExit(f"VideoREPA root not found: {vr}")
        run_videophy2()
    elif args.which == "vbench":
        run_vbench()


if __name__ == "__main__":
    main()
