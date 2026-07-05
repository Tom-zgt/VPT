import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VALID_ROLES = {"agent", "agent_controlled_object", "passive_object"}
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

logger = logging.getLogger(__name__)


@dataclass
class Category:
    name: str
    role: str
    reason: str


def load_qwen_model(
    model_id: str = DEFAULT_MODEL_ID,
    attn_implementation: Optional[str] = "sdpa",
):
    """
    Load Qwen3-VL model and processor.

    This follows the official HF README signature. Keep `trust_remote_code=True`
    because Qwen ships custom generate helpers.
    """
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "transformers is required for Qwen inference. Install with "
            'pip install "transformers>=4.57.0" accelerate einops tiktoken'
        ) from exc

    kwargs: Dict[str, Any] = {
        "dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except ImportError as exc:
        msg = str(exc)
        if "flash_attn" in msg and attn_implementation != "sdpa":
            logger.warning("FlashAttention not available, retrying with sdpa.")
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        else:
            raise
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def _load_video_frames(video_path: str) -> List[np.ndarray]:
    """Fallback loader when qwen_vl_utils is unavailable."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
    return frames


def _extract_json(text: str) -> Dict[str, Any]:
    """Pick the first JSON object from the model output."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"Qwen output does not contain JSON: {text}")
    candidate = match.group(0)
    # Normalize common full-width quotes if present
    candidate = candidate.replace("“", '"').replace("”", '"')
    return json.loads(candidate)


def _normalize_categories(payload: Dict[str, Any]) -> List[Category]:
    categories_raw = payload.get("categories", [])
    cleaned: List[Category] = []
    for item in categories_raw:
        name = str(item.get("name", "")).strip()
        role = str(item.get("role", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not name or role not in VALID_ROLES:
            continue
        cleaned.append(Category(name=name.lower(), role=role, reason=reason))
    if not cleaned:
        raise ValueError("No valid categories parsed from Qwen output.")
    return cleaned


def infer_categories(
    model,
    processor,
    video_path: str,
    video_prompt: str,
    system_prompt_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    max_new_tokens: int = 512,
    attn_implementation: Optional[str] = "flash_attention_2",
) -> Tuple[List[Category], Dict[str, Any]]:
    """
    Run Qwen3-VL-30B-A3B on a video and return structured categories.

    Returns:
        categories: parsed Category list
        raw: a dict with the raw_text and messages used
    """
    system_prompt = Path(system_prompt_path).read_text(encoding="utf-8")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)


    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{os.path.abspath(video_path)}",
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": 256 * 32 * 32,
                    "total_pixels": 20480 * 32 * 32,
                    # "total_pixels": 2560 * 32 * 32,
                },
                {
                    "type": "text",
                    "text": "Return JSON with categories, role, and reason per the system prompt.",
                },
            ],
        },
    ]

    messages_small: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "system", "content": [{"type": "text", "text": "this is the prompt of this video: " + video_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{os.path.abspath(video_path)}",
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": 256 * 32 * 32,
                    # "total_pixels": 20480 * 32 * 32,
                    "total_pixels": 2560 * 32 * 32,
                },
                {
                    "type": "text",
                    "text": "Return JSON with categories, role, and reason per the system prompt.",
                },
            ],
        },
    ]

    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception:
        process_vision_info = None
    try:
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if process_vision_info:
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True
            )
            if videos is not None:
                videos = [v[0] if isinstance(v, (list, tuple)) else v for v in videos]
            model_inputs = processor(
                text=[text_prompt],
                images=images,
                videos=videos,
                video_details=video_kwargs,
                return_tensors="pt",
            ).to(model.device)
        else:
            frames = _load_video_frames(video_path)
            model_inputs = processor(
                text=[text_prompt], videos=[frames], return_tensors="pt"
            ).to(model.device).to(model.dtype)
            logger.info("process_vision_info not found; falling back to manual video loader.")

        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    except Exception:
        
        text_prompt = processor.apply_chat_template(
            messages_small, tokenize=False, add_generation_prompt=True
        )

        if process_vision_info:
            images, videos, video_kwargs = process_vision_info(
                messages_small, image_patch_size=16, return_video_kwargs=True
            )
            if videos is not None:
                videos = [v[0] if isinstance(v, (list, tuple)) else v for v in videos]
            model_inputs = processor(
                text=[text_prompt],
                images=images,
                videos=videos,
                video_details=video_kwargs,
                return_tensors="pt",
            ).to(model.device)
        else:
            frames = _load_video_frames(video_path)
            print(len(frames))
            model_inputs = processor(
                text=[text_prompt], videos=[frames], return_tensors="pt"
            ).to(model.device).to(model.dtype)
            logger.info("process_vision_info not found; falling back to manual video loader.")

        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens//2)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    payload = _extract_json(output_text)
    categories = _normalize_categories(payload)
    return categories, {"raw_text": output_text, "messages": messages}
