import numpy as np
from typing import Dict, List, Tuple

ROLE_AGENT = "agent"
ROLE_AGENT_CTRL = "agent_controlled_object"
ROLE_PASSIVE = "passive_object"
VALID_ROLES = {ROLE_AGENT, ROLE_AGENT_CTRL, ROLE_PASSIVE}


def _to_numpy(arr):
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


def aggregate_role_masks(
    outputs_per_frame: Dict[int, Dict],
    obj_id_to_role: Dict[int, str],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Convert SAM3 outputs to per-role binary masks.

    Args:
        outputs_per_frame: {frame_idx: {"out_obj_ids": tensor, "out_binary_masks": tensor}}
        obj_id_to_role: {obj_id: role}
    """
    role_masks_per_frame: Dict[int, Dict[str, np.ndarray]] = {}
    for frame_idx, outputs in outputs_per_frame.items():
        obj_ids = _to_numpy(outputs["out_obj_ids"])
        masks = _to_numpy(outputs["out_binary_masks"])
        if masks.ndim == 2:
            masks = masks[None, ...]

        H, W = masks.shape[-2:]
        frame_roles = {
            ROLE_AGENT: np.zeros((H, W), dtype=bool),
            ROLE_AGENT_CTRL: np.zeros((H, W), dtype=bool),
            ROLE_PASSIVE: np.zeros((H, W), dtype=bool),
        }

        for obj_id, mask in zip(obj_ids, masks):
            role = obj_id_to_role.get(int(obj_id))
            if role not in VALID_ROLES:
                continue
            frame_roles[role] |= mask.astype(bool)

        role_masks_per_frame[frame_idx] = frame_roles
    return role_masks_per_frame


def build_semantic_channel(role_masks: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Encode masks into a single semantic channel with uint8 values in {0, 85, 170, 255}.
    """
    shape = next(iter(role_masks.values())).shape
    S = np.zeros(shape, dtype=np.uint8)
    if role_masks[ROLE_PASSIVE].any():
        S[role_masks[ROLE_PASSIVE]] = 85  # 255 * (1/3)
    if role_masks[ROLE_AGENT_CTRL].any():
        S[role_masks[ROLE_AGENT_CTRL]] = 170  # 255 * (2/3)
    if role_masks[ROLE_AGENT].any():
        S[role_masks[ROLE_AGENT]] = 255
    return S


def compose_four_channel_frame(frame_rgb: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """
    Concatenate RGB and semantic channel -> [H, W, 4].
    """
    if semantic.ndim != 2:
        raise ValueError("Semantic channel must be HxW.")
    if frame_rgb.shape[:2] != semantic.shape:
        raise ValueError("RGB frame and semantic channel must share spatial shape.")
    semantic_expanded = semantic[..., None].astype(np.uint8)
    return np.concatenate([frame_rgb.astype(np.uint8), semantic_expanded], axis=-1)
