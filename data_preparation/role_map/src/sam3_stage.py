# import logging
# from typing import Dict, Iterable, List, Optional, Tuple

# import numpy as np
# import cv2

# logger = logging.getLogger(__name__)


# def segment_video_with_sam3(
#     predictor,
#     video_path: str,
#     text_prompts: Iterable[str],
#     propagation_direction: str = "both",
#     gpus_to_use: Optional[List[int]] = None,
# ):
#     """
#     Run SAM3 dense video tracking with text prompts.

#     For each prompt, independently find a frame that produces a non-empty mask,
#     propagate from that frame, and merge all outputs.

#     Returns:
#         outputs_per_frame: {frame_idx: output_dict}
#         obj_id_to_text: mapping for downstream role assignment
#     """
#     try:
#         from sam3.model_builder import build_sam3_video_predictor
#     except ModuleNotFoundError as exc:  # pragma: no cover - import guard
#         missing = str(exc)
#         if "decord" in missing:
#             raise RuntimeError(
#                 "Missing dependency decord. Install with: pip install decord "
#                 "or pip install -e '.[notebooks]' inside the sam3 repo."
#             ) from exc
#         raise RuntimeError(
#             "sam3 is required. Clone https://github.com/facebookresearch/sam3 and pip install -e . "
#             "Also ensure optional deps like decord are installed for video."
#         ) from exc

#     # Determine candidate frames for prompting (spread across the video)
#     candidate_frames: List[int] = [0]
#     total_frames = None
#     try:
#         cap = cv2.VideoCapture(video_path)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         cap.release()
#         if total_frames and total_frames > 0:
#             stride = max(1, total_frames // 10)  # ~10 candidates across the clip
#             candidate_frames = list(range(0, total_frames, stride))
#             candidate_frames.append(max(0, total_frames - 1))
#             candidate_frames = sorted(set(candidate_frames))
#     except Exception:
#         pass

    
#     obj_id_to_text: Dict[int, str] = {}
#     merged_outputs: Dict[int, Dict] = {}

#     def _to_numpy(x):
#         if x is None:
#             return None
#         try:
#             if hasattr(x, "cpu"):
#                 x = x.cpu()
#             if hasattr(x, "numpy"):
#                 return x.numpy()
#         except Exception:
#             pass
#         return np.asarray(x)

#     def merge_outputs(new_outputs: Dict[int, Dict], target_obj_id: int):
#         for fidx, out in new_outputs.items():
#             masks = out.get("out_binary_masks")
#             if masks is None:
#                 continue
#             masks_np = _to_numpy(masks)
#             if masks_np is None or masks_np.size == 0:
#                 continue
#             obj_ids_arr = np.full((masks_np.shape[0],), target_obj_id, dtype=int)

#             entry = merged_outputs.get(fidx)
#             if entry is None:
#                 merged_outputs[fidx] = {
#                     "out_binary_masks": masks_np,
#                     "out_obj_ids": obj_ids_arr,
#                     "out_probs": _to_numpy(out.get("out_probs")),
#                     "out_boxes_xywh": _to_numpy(out.get("out_boxes_xywh")),
#                     "frame_stats": out.get("frame_stats"),
#                 }
#             else:
#                 # append to existing
#                 try:
#                     merged_masks = _to_numpy(entry.get("out_binary_masks"))
#                     merged_ids = _to_numpy(entry.get("out_obj_ids"))
#                     merged_outputs[fidx]["out_binary_masks"] = (
#                         np.concatenate([merged_masks, masks_np], axis=0)
#                         if merged_masks is not None
#                         else masks_np
#                     )
#                     merged_outputs[fidx]["out_obj_ids"] = (
#                         np.concatenate([merged_ids, obj_ids_arr], axis=0)
#                         if merged_ids is not None
#                         else obj_ids_arr
#                     )
#                 except Exception:
#                     merged_outputs[fidx]["out_binary_masks"] = masks_np
#                     merged_outputs[fidx]["out_obj_ids"] = obj_ids_arr

#     try:
#         for idx, prompt in enumerate(text_prompts):
#             obj_id = idx + 1
#             obj_id_to_text[obj_id] = prompt
#             response = predictor.handle_request(
#                 {"type": "start_session", "resource_path": video_path}
#             )
#             session_id = response["session_id"]

#             chosen_frame_index = None
#             for fi in candidate_frames:
#                 try:
#                     resp = predictor.handle_request(
#                         {
#                             "type": "add_prompt",
#                             "session_id": session_id,
#                             "frame_index": fi,
#                             "text": prompt,
#                             "obj_id": obj_id,
#                         }
#                     )
#                 except Exception as exc:
#                     logger.warning(
#                         "SAM3 add_prompt failed on frame %s for '%s': %s", fi, prompt, exc
#                     )
#                     continue
#                 out = resp.get("outputs", {})
#                 masks = out.get("out_binary_masks")
#                 has_mask = False
#                 try:
#                     has_mask = masks is not None and np.asarray(masks).any()
#                 except Exception:
#                     has_mask = False
#                 if has_mask:
#                     chosen_frame_index = fi
#                     break
#             if chosen_frame_index is None:
#                 logger.warning("SAM3 add_prompt produced no mask for '%s'", prompt)
#                 try:
#                     predictor.handle_request({"type": "close_session", "session_id": session_id})
#                 except Exception:
#                     pass
#                 continue

#             # propagate for this prompt/session
#             per_prompt_outputs: Dict[int, Dict] = {}
#             for resp in predictor.handle_stream_request(
#                 {
#                     "type": "propagate_in_video",
#                     "session_id": session_id,
#                     "propagation_direction": propagation_direction,
#                     "start_frame_index": chosen_frame_index,
#                 }
#             ):
#                 per_prompt_outputs[resp["frame_index"]] = resp["outputs"]

#             merge_outputs(per_prompt_outputs, target_obj_id=obj_id)

#             # close session for this prompt
#             try:
#                 predictor.handle_request({"type": "close_session", "session_id": session_id})
#             except Exception:
#                 pass

#     finally:
#         try:
#             predictor.shutdown()
#         except Exception:
#             pass

#     return merged_outputs, obj_id_to_text
import logging
import gc
import torch
import numpy as np
import cv2
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

def segment_video_with_sam3(
    predictor,
    video_path: str,
    text_prompts: Iterable[str],
    propagation_direction: str = "both",
    gpus_to_use: Optional[List[int]] = None,
):
    """
    Run SAM3 dense video tracking with text prompts.
    Optimized to use a SINGLE session for all prompts to avoid OOM.
    """
    # 依赖检查保持不变
    try:
        from sam3.model_builder import build_sam3_video_predictor
    except ModuleNotFoundError as exc:
        if "decord" in str(exc):
            raise RuntimeError("Missing dependency decord.") from exc
        raise RuntimeError("sam3 is required.") from exc

    # 1. 确定候选帧 (保持不变)
    candidate_frames: List[int] = [0]
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames and total_frames > 0:
            stride = max(1, total_frames // 10)
            candidate_frames = list(range(0, total_frames, stride))
            candidate_frames.append(max(0, total_frames - 1))
            candidate_frames = sorted(set(candidate_frames))
    except Exception:
        pass

    obj_id_to_text: Dict[int, str] = {}
    merged_outputs: Dict[int, Dict] = {}

    # 辅助函数：转 Numpy
    def _to_numpy(x):
        if x is None: return None
        try:
            if hasattr(x, "cpu"): x = x.cpu()
            if hasattr(x, "numpy"): return x.numpy()
        except Exception: pass
        return np.asarray(x)

    # 显式清理缓存，防止上一次调用的残留
    gc.collect()
    torch.cuda.empty_cache()

    # 【关键修改 1】在循环外部开启 Session，只加载一次视频
    logger.info(f"Loading video into SAM3 session: {video_path}")
    try:
        response = predictor.handle_request(
            {"type": "start_session", "resource_path": video_path}
        )
        session_id = response["session_id"]
    except Exception as e:
        logger.error(f"Failed to start SAM3 session: {e}")
        # 如果开启失败，可能是显存真不够了，再次尝试清理后报错
        torch.cuda.empty_cache()
        raise e

    try:
        active_obj_ids = []
        
        # 【关键修改 2】循环添加 Prompt 到同一个 Session
        for idx, prompt in enumerate(text_prompts):
            obj_id = idx + 1
            obj_id_to_text[obj_id] = prompt
            
            chosen_frame_index = None
            
            # 在当前 Session 中寻找该 Prompt 的最佳帧
            for fi in candidate_frames:
                try:
                    resp = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": fi,
                            "text": prompt,
                            "obj_id": obj_id,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"SAM3 add_prompt failed frame {fi} for '{prompt}': {exc}")
                    continue

                out = resp.get("outputs", {})
                masks = out.get("out_binary_masks")
                
                # 检查是否有有效 Mask
                has_mask = False
                try:
                    if masks is not None:
                        # 这是一个 Tensor 或 Array，检查是否有 True
                        if hasattr(masks, "any"):
                            has_mask = masks.any()
                        else:
                            has_mask = np.asarray(masks).any()
                except Exception:
                    pass
                
                if has_mask:
                    chosen_frame_index = fi
                    break
            
            if chosen_frame_index is not None:
                active_obj_ids.append(obj_id)
                logger.info(f"Prompt '{prompt}' (id={obj_id}) added at frame {chosen_frame_index}")
            else:
                logger.warning(f"SAM3 could not find mask for prompt '{prompt}'")

        # 【关键修改 3】一次性对所有对象进行传播 (Propagation)
        # SAM3 支持多对象同时追踪，这样非常节省显存且速度快
        if active_obj_ids:
            logger.info(f"Propagating {len(active_obj_ids)} objects in video...")
            
            # 注意：如果多个 prompt 在不同帧初始化的，propagate_in_video 通常能处理
            # 但为了保险，有时需要指定 start_frame_index。
            # 如果是 "both" 方向，通常不需要指定 start_frame_index，或者它会自动处理全视频。
            # 这里我们不传 start_frame_index，让 SAM3 自动处理已添加的关键帧。
            
            stream_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
            }

            for resp in predictor.handle_stream_request(stream_request):
                fidx = resp["frame_index"]
                out = resp["outputs"]
                
                # SAM3 的输出直接包含该帧所有追踪到的对象
                # out["out_obj_ids"] 是一个列表，包含当前帧存在的对象 ID
                # out["out_binary_masks"] 是对应的 masks
                
                # 我们直接转换并存储，不再需要手动的 merge_outputs 拼接逻辑
                # 因为 SAM3 已经在内部帮我们合并好了这一帧的所有对象
                
                if out.get("out_binary_masks") is None:
                    continue

                # 存入字典
                merged_outputs[fidx] = {
                    "out_binary_masks": _to_numpy(out.get("out_binary_masks")),
                    "out_obj_ids": _to_numpy(out.get("out_obj_ids")),
                    "out_probs": _to_numpy(out.get("out_probs")),
                    "out_boxes_xywh": _to_numpy(out.get("out_boxes_xywh")),
                    "frame_stats": out.get("frame_stats"),
                }
        else:
            logger.warning("No active objects to propagate.")

    except Exception as e:
        logger.error(f"Error during SAM3 segmentation: {e}")
        raise e

    finally:
        # 【关键修改 4】清理 Session 和显存
        try:
            # 关闭 Session，释放视频 Tensor
            if 'session_id' in locals():
                predictor.handle_request({"type": "close_session", "session_id": session_id})
        except Exception:
            pass
        
        # 移除 predictor.shutdown()！
        # 原因：predictor 是从外部传入的。如果在这里 shutdown，下一次调用函数时 predictor 就失效了。
        # 应该由外部调用者（生命周期管理者）决定何时 shutdown。
        
        # 强制 GC 和清空显存
        gc.collect()
        torch.cuda.empty_cache()

    return merged_outputs, obj_id_to_text