import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from src import mask_ops
from src.qwen_stage import infer_categories
from src.sam3_stage import segment_video_with_sam3

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Failed to read video metadata: {video_path}")
    return fps, frame_count, height, width


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline_rgb(
    video_path: str,
    system_prompt_path: str,
    output_dir: str,
    model_id: str,
    max_new_tokens: int,
):
    video_path = os.path.abspath(video_path)
    stem = Path(video_path).stem
    output_root = Path(output_dir)
    metadata_dir = output_root / "metadata"
    rgb_dir = output_root / "rgb"
    for d in [metadata_dir, rgb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Stage A: Qwen3-VL categorization
    logger.info("Running Qwen3-VL on %s ...", video_path)
    categories, raw = infer_categories(
        video_path=video_path,
        system_prompt_path=system_prompt_path,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
    )
    cat_json = [
        {"name": c.name, "role": c.role, "reason": c.reason} for c in categories
    ]
    save_json(
        {"categories": cat_json, "raw_text": raw["raw_text"]},
        metadata_dir / f"{stem}_categories.json",
    )
    logger.info("Parsed %d categories: %s", len(categories), [c.name for c in categories])

    text_prompts = [c.name for c in categories]
    hardcoded_roles = {
        1: "agent",
        2: "agent_controlled_object",
        3: "passive_object",
    }
    obj_id_to_role: Dict[int, str] = {
        idx + 1: hardcoded_roles.get(idx + 1, "passive_object")
        for idx, _ in enumerate(categories)
    }

    # Stage B: SAM3 segmentation
    logger.info("Running SAM3 propagation ...")
    outputs_per_frame, _ = segment_video_with_sam3(
        video_path=video_path, text_prompts=text_prompts
    )
    seen_ids = set()
    if outputs_per_frame:
        for out in outputs_per_frame.values():
            obj_ids = out.get("out_obj_ids")
            if obj_ids is not None:
                try:
                    arr = obj_ids.cpu().numpy() if hasattr(obj_ids, "cpu") else np.asarray(obj_ids)
                    seen_ids.update(int(x) for x in arr.tolist())
                except Exception:
                    pass
        if seen_ids:
            logger.info("SAM3 obj_ids seen: %s", sorted(seen_ids))
    else:
        logger.warning("SAM3 returned no outputs; RGB mask video will be empty.")

    if seen_ids:
        first_role = obj_id_to_role.get(1, "agent")
        for oid in seen_ids:
            if oid not in obj_id_to_role:
                obj_id_to_role[oid] = first_role
                logger.info("Mapping SAM3 obj_id %s to role '%s' to keep its masks.", oid, first_role)

    role_masks_per_frame = mask_ops.aggregate_role_masks(outputs_per_frame, obj_id_to_role)

    # Stage C: build 3-channel semantic RGB map
    fps, frame_count, video_h, video_w = get_video_meta(video_path)
    semantic_rgb_frames: List[np.ndarray] = []

    if role_masks_per_frame:
        sample_masks = next(iter(role_masks_per_frame.values()))
        default_shape = next(iter(sample_masks.values())).shape
    else:
        default_shape = (video_h, video_w)

    for idx in range(frame_count):
        role_masks = role_masks_per_frame.get(idx)
        if role_masks is None:
            H, W = default_shape
            role_masks = {
                mask_ops.ROLE_AGENT: np.zeros((H, W), dtype=bool),
                mask_ops.ROLE_AGENT_CTRL: np.zeros((H, W), dtype=bool),
                mask_ops.ROLE_PASSIVE: np.zeros((H, W), dtype=bool),
            }
        agent = role_masks[mask_ops.ROLE_AGENT]
        agent_ctrl = role_masks[mask_ops.ROLE_AGENT_CTRL]
        passive = role_masks[mask_ops.ROLE_PASSIVE]
        bg = ~(agent | agent_ctrl | passive)

        sem = np.zeros((*bg.shape, 3), dtype=np.uint8)
        sem[bg] = (0, 0, 0)
        sem[passive] = (85, 85, 85)
        sem[agent_ctrl] = (170, 170, 170)
        sem[agent] = (255, 255, 255)
        semantic_rgb_frames.append(sem)

    # Save as mp4 (grayscale-like but 3-channel)
    semantic_video_path = rgb_dir / f"{stem}_rgb.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    H, W = semantic_rgb_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(semantic_video_path), fourcc, fps, (W, H), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {semantic_video_path}")
    for frame in semantic_rgb_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    logger.info("Saved RGB mask video to %s", semantic_video_path)


def iter_mp4_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"]
        return sorted(files)
    raise FileNotFoundError(input_path)


def build_argparser():
    parser = argparse.ArgumentParser(description="Semantic RGB mask pipeline")
    parser.add_argument("--video", required=True, help="Path to input video (mp4).")
    parser.add_argument(
        "--system-prompt",
        default="prompts/qwen_system.txt",
        help="Path to system prompt file.",
    )
    parser.add_argument(
        "--output-dir", default="data/outputs", help="Directory for all outputs."
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Hugging Face model id for Qwen3-VL.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Generation length for Qwen."
    )
    return parser


def main():
    args = build_argparser().parse_args()
    input_path = Path(args.video)
    video_paths = iter_mp4_files(input_path)
    if input_path.is_dir():
        logger.info("Found %d mp4 files under %s", len(video_paths), input_path)
    if not video_paths:
        raise RuntimeError(f"No mp4 files found under: {input_path}")
    for vp in video_paths:
        run_pipeline_rgb(
            video_path=str(vp),
            system_prompt_path=args.system_prompt,
            output_dir=args.output_dir,
            model_id=args.model_id,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
