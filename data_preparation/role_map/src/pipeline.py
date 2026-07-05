import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import subprocess

from src import mask_ops
from src.qwen_stage import Category, infer_categories
from src.sam3_stage import segment_video_with_sam3

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_frames_rgb(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
    return frames, fps


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
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
    semantic_dir = output_root / "semantic"
    for d in [metadata_dir, semantic_dir]:
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

    # Build prompt list and role map
    text_prompts = [c.name for c in categories]
    # Hardcoded role mapping by prompt index
    hardcoded_roles = {
        1: "agent",
        2: "agent_controlled_object",
        3: "passive_object",
    }
    obj_id_to_role: Dict[int, str] = {idx + 1: hardcoded_roles.get(idx + 1, "passive_object") for idx, _ in enumerate(categories)}

    # Stage B: SAM3 segmentation
    logger.info("Running SAM3 propagation ...")
    outputs_per_frame, _ = segment_video_with_sam3(
        video_path=video_path, text_prompts=text_prompts
    )
    seen_ids = set()
    if not outputs_per_frame:
        logger.warning("SAM3 returned no outputs; semantic video will be empty.")
    else:
        frames_with_masks = []
        first_mask_dump = None
        first_raw_dump = None
        for frame_idx, out in outputs_per_frame.items():
            obj_ids = out.get("out_obj_ids")
            masks = out.get("out_binary_masks")
            if obj_ids is not None:
                try:
                    arr = obj_ids.cpu().numpy() if hasattr(obj_ids, "cpu") else np.asarray(obj_ids)
                    seen_ids.update(int(x) for x in arr.tolist())
                except Exception:
                    pass
            has_mask = False
            if masks is not None:
                try:
                    has_mask = np.asarray(masks).any()
                except Exception:
                    has_mask = False
            if has_mask:
                frames_with_masks.append(frame_idx)
                if first_mask_dump is None:
                    first_mask_dump = {
                        "frame_idx": frame_idx,
                        "obj_ids": obj_ids.cpu().numpy()
                        if hasattr(obj_ids, "cpu")
                        else np.asarray(obj_ids),
                        "masks": masks.cpu().numpy()
                        if hasattr(masks, "cpu")
                        else np.asarray(masks),
                    }
                    first_raw_dump = out

        frames_with_masks = sorted(frames_with_masks)
        logger.info(
            "SAM3 frames: %d, frames with any mask: %d",
            len(outputs_per_frame),
            len(frames_with_masks),
        )
        if seen_ids:
            logger.info("SAM3 obj_ids seen: %s", sorted(seen_ids))
        if frames_with_masks:
            logger.info(
                "First frame with mask: %d; last: %d",
                frames_with_masks[0],
                frames_with_masks[-1],
            )
        if first_mask_dump is not None:
            debug_path = Path(output_dir) / "semantic" / f"{stem}_sam3_debug.npz"
            np.savez_compressed(
                debug_path,
                frame_idx=first_mask_dump["frame_idx"],
                obj_ids=first_mask_dump["obj_ids"],
                masks=first_mask_dump["masks"],
            )
            logger.info("Saved SAM3 debug masks for first frame to %s", debug_path)
        if first_raw_dump is not None:
            raw_path = Path(output_dir) / "semantic" / f"{stem}_sam3_raw.npz"
            np.savez_compressed(raw_path, raw=np.array(first_raw_dump, dtype=object))
            logger.info("Saved SAM3 raw outputs for first frame to %s", raw_path)

    # Map any unseen obj_ids to the first role so masks are not dropped
    if seen_ids:
        first_role = obj_id_to_role.get(1, categories[0].role if categories else "agent")
        for oid in seen_ids:
            if oid not in obj_id_to_role:
                obj_id_to_role[oid] = first_role
                logger.info("Mapping SAM3 obj_id %s to role '%s' to keep its masks.", oid, first_role)

    role_masks_per_frame = mask_ops.aggregate_role_masks(outputs_per_frame, obj_id_to_role)

    # Stage C: semantic channel only (uint8 grayscale)
    frames_rgb, fps = load_frames_rgb(video_path)
    semantic_frames: List[np.ndarray] = []

    if role_masks_per_frame:
        sample_masks = next(iter(role_masks_per_frame.values()))
        default_shape = next(iter(sample_masks.values())).shape
    else:
        H, W = frames_rgb[0].shape[:2]
        default_shape = (H, W)

    for idx in range(len(frames_rgb)):
        role_masks = role_masks_per_frame.get(idx)
        if role_masks is None:
            H, W = default_shape
            role_masks = {
                mask_ops.ROLE_AGENT: np.zeros((H, W), dtype=bool),
                mask_ops.ROLE_AGENT_CTRL: np.zeros((H, W), dtype=bool),
                mask_ops.ROLE_PASSIVE: np.zeros((H, W), dtype=bool),
            }
        semantic = mask_ops.build_semantic_channel(role_masks)
        semantic_frames.append(semantic.astype(np.uint8))

    # Write semantic video using OpenCV (gray, single channel)
    semantic_video_path = semantic_dir / f"{stem}_S.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    H, W = semantic_frames[0].shape
    writer = cv2.VideoWriter(str(semantic_video_path), fourcc, fps, (W, H), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {semantic_video_path}")
    for frame in semantic_frames:
        writer.write(frame)
    writer.release()
    logger.info("Saved semantic channel video to %s", semantic_video_path)


def build_argparser():
    parser = argparse.ArgumentParser(description="Qwen3-VL + SAM3 video pipeline")
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
    run_pipeline(
        video_path=args.video,
        system_prompt_path=args.system_prompt,
        output_dir=args.output_dir,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
