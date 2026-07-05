import argparse
import json
import logging
import os
import traceback  # 新增: 用于打印详细报错堆栈
from pathlib import Path
from typing import Dict, List
import subprocess  # 新增：用于调用 ffmpeg 命令行
import torch
import cv2
import numpy as np
from pathlib import Path
from src import mask_ops
from src.qwen_stage import infer_categories, load_qwen_model
from src.sam3_stage import segment_video_with_sam3
from tqdm import tqdm
import pandas as pd
import gc
from sam3.model_builder import build_sam3_video_predictor
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def check_sam_params_exists(src_path_a, dir_b):
    """
    输入: 
      src_path_a: /pathA/xxx.mp4 (原文件的完整路径)
      dir_b:      /pathB/        (要去查找的目标目录)
    返回: 
      True/False
    """
    # 1. 把路径转为 Path 对象
    src = Path(src_path_a)
    target_dir = Path(dir_b)
    
    # 2. 构建目标文件名: xxx -> xxx_rgb.mp4
    # .stem 是文件名(无后缀)，.suffix 是后缀(.mp4)
    target_name = f"{src.stem}_sam_params.json"
    
    # 3. 拼接到 pathB: /pathB/ + xxx_rgb.mp4
    target_path = (target_dir / 'sam_param') / target_name
    
    # 4. 检查是否存在
    return target_path.exists()

def check_rgb_exists(src_path_a, dir_b):
    """
    输入: 
      src_path_a: /pathA/xxx.mp4 (原文件的完整路径)
      dir_b:      /pathB/        (要去查找的目标目录)
    返回: 
      True/False
    """
    # 1. 把路径转为 Path 对象
    src = Path(src_path_a)
    target_dir = Path(dir_b)
    
    # 2. 构建目标文件名: xxx -> xxx_rgb.mp4
    # .stem 是文件名(无后缀)，.suffix 是后缀(.mp4)
    target_name = f"{src.stem}_rgb{src.suffix}"
    
    # 3. 拼接到 pathB: /pathB/ + xxx_rgb.mp4
    target_path = (target_dir / 'rgb') / target_name
    print(target_path)
    # 4. 检查是否存在
    return target_path.exists()

def save_video_with_ffmpeg(frames: List[np.ndarray], output_path: str, fps: float):
    """
    使用 ffmpeg 命令行直接管道输入保存视频，避开 OpenCV VideoWriter 的坑。
    """
    if not frames:
        logger.warning("No frames to save.")
        return

    # 获取尺寸
    h, w = frames[0].shape[:2]
    output_path = str(output_path)

    # 构建 FFmpeg 命令
    # -y: 覆盖输出文件
    # -f rawvideo: 输入格式为原始视频流
    # -vcodec rawvideo: 输入编码
    # -s {w}x{h}: 帧尺寸
    # -pix_fmt rgb24: 输入像素格式 (因为你的 semantic_rgb_frames 是 RGB)
    # -r {fps}: 帧率
    # -i -: 从标准输入读取数据
    # -c:v libx264: 输出编码器 H.264
    # -pix_fmt yuv420p: 输出像素格式 (兼容所有播放器)
    # -preset fast: 编码速度
    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', f'{fps}',
        '-i', '-', 
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '18', # 高质量
        output_path
    ]

    logger.info(f"FFmpeg command: {' '.join(cmd)}")
    
    process = None
    try:
        # 启动 ffmpeg 进程，准备接收数据
        process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL
        )

        for frame in frames:
            # 确保数据类型正确
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # 确保尺寸一致 (鲁棒性检查)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)

            # 将 numpy 数组转为字节流写入管道
            process.stdin.write(frame.tobytes())

        # 完成写入
        process.stdin.close()
        process.wait()
        
        if process.returncode != 0:
            # 如果出错，读取错误日志
            _, stderr = process.communicate()
            logger.error(f"FFmpeg Error:\n{stderr.decode('utf-8')}")
            raise RuntimeError("FFmpeg failed to write video.")
            
        logger.info(f"Saved RGB mask video to {output_path}")

    except FileNotFoundError:
        logger.error("FFmpeg command not found. Please install ffmpeg (apt install ffmpeg).")
        raise
    except Exception as e:
        logger.error(f"Error during ffmpeg pipe: {e}")
        if process:
            process.kill()
        raise
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
    model, processor, predictor,
    video_path: str,
    video_prompt: str,
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
        model, processor,
        video_path=video_path,
        video_prompt=video_prompt,
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
        predictor,
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
    # semantic_video_path = rgb_dir / f"{stem}_rgb.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # # fourcc = cv2.VideoWriter_fourcc(*"avc1")
    # # fourcc = cv2.VideoWriter_fourcc(*"H264")
    # H, W = semantic_rgb_frames[0].shape[:2]
    # writer = cv2.VideoWriter(str(semantic_video_path), fourcc, fps, (W, H), isColor=True)
    # if not writer.isOpened():
    #     raise RuntimeError(f"Failed to open VideoWriter for {semantic_video_path}")
    # for frame in semantic_rgb_frames:
    #     # writer.write(frame)
    #     writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # writer.release()
    semantic_video_path = rgb_dir / f"{stem}_rgb.mp4"
    save_video_with_ffmpeg(
        frames=semantic_rgb_frames,
        output_path=str(semantic_video_path),
        fps=fps
    )
    logger.info("Saved RGB mask video to %s", semantic_video_path)

def run_pipeline_und(
    model, processor, predictor,
    video_path: str,
    video_prompt: str,
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
    sam_param_dir = output_root / "sam_param"
    for d in [metadata_dir, rgb_dir, sam_param_dir]:
        d.mkdir(parents=True, exist_ok=True)
    sam_params_path = sam_param_dir / f"{stem}_sam_params.json"
    print(sam_params_path)
    if sam_params_path.exists():
        logger.info(f"Skipping Qwen inference, found existing params: {sam_params_path}")
        try:
            data = json.loads(sam_params_path.read_text(encoding="utf-8"))
            # JSON key 是字符串，需要转回 int
            obj_id_to_role = {int(k): v for k, v in data["obj_id_to_role"].items()}
            text_prompts = data["text_prompts"]
            return obj_id_to_role, text_prompts
        except Exception as e:
            logger.warning(f"Failed to load existing params, re-running Qwen: {e}")
    # Stage A: Qwen3-VL categorization
    logger.info("Running Qwen3-VL on %s ...", video_path)
    categories, raw = infer_categories(
        model, processor,
        video_path=video_path,
        video_prompt=video_prompt,
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
    # --- 新增：保存 SAM 需要的中间参数 ---
    # 将 obj_id_to_role 和 text_prompts 保存下来
    save_data = {
        "obj_id_to_role": obj_id_to_role,  # JSON dump 会自动把 int key 转为 string
        "text_prompts": text_prompts
    }
    save_json(save_data, sam_params_path)
    logger.info(f"Saved intermediate SAM params to {sam_params_path}")
    return obj_id_to_role, text_prompts

def run_pipeline_mask(
    model, processor, predictor1,
    video_path: str,
    video_prompt: str,
    system_prompt_path: str,
    output_dir: str,
    model_id: str,
    max_new_tokens: int,
    obj_id_to_role, text_prompts,
):
    predictor = build_sam3_video_predictor()
    video_path = os.path.abspath(video_path)
    stem = Path(video_path).stem
    output_root = Path(output_dir)
    metadata_dir = output_root / "metadata"
    rgb_dir = output_root / "rgb"
    # Stage B: SAM3 segmentation
    logger.info("Running SAM3 propagation ...")
    outputs_per_frame, _ = segment_video_with_sam3(
        predictor,
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
    del outputs_per_frame
    torch.cuda.empty_cache() # 强制释放 PyTorch 缓存
    # Stage C: build 3-channel semantic RGB map
    # fps, frame_count, video_h, video_w = get_video_meta(video_path)
    # semantic_rgb_frames: List[np.ndarray] = []
    fps, frame_count, video_h, video_w = get_video_meta(video_path)
    semantic_video_path = rgb_dir / f"{stem}_rgb.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(semantic_video_path), fourcc, fps, (video_w, video_h), isColor=True)
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {semantic_video_path}")

    # 确定默认 shape
    default_shape = (video_h, video_w)
    if role_masks_per_frame:
        try:
            sample_masks = next(iter(role_masks_per_frame.values()))
            # 确保获取正确的形状
            first_mask = next(iter(sample_masks.values()))
            default_shape = first_mask.shape
        except StopIteration:
            pass

    # 逐帧生成并写入，内存用完即丢
    for idx in range(frame_count):
        role_masks = role_masks_per_frame.get(idx)
        H, W = default_shape # 使用之前确定的 shape
        
        if role_masks is None:
            # 构造空 mask
            agent = np.zeros((H, W), dtype=bool)
            agent_ctrl = np.zeros((H, W), dtype=bool)
            passive = np.zeros((H, W), dtype=bool)
        else:
            agent = role_masks.get(mask_ops.ROLE_AGENT, np.zeros((H, W), dtype=bool))
            agent_ctrl = role_masks.get(mask_ops.ROLE_AGENT_CTRL, np.zeros((H, W), dtype=bool))
            passive = role_masks.get(mask_ops.ROLE_PASSIVE, np.zeros((H, W), dtype=bool))

        bg = ~(agent | agent_ctrl | passive)

        # 构建图像 (H, W, 3)
        sem = np.zeros((H, W, 3), dtype=np.uint8)
        # BGR 格式 (OpenCV 默认是 BGR)
        # 如果你想要 RGB，需要注意 cv2.cvtColor 或者直接在这里赋值 BGR 颜色
        # 假设这里依然用原本的逻辑，但注意 cv2 写盘需要 BGR
        
        # 你的原始代码是 RGB 定义: (0,0,0), (85,85,85), (170..), (255..)
        # 因为是灰度色阶，RGB和BGR是一样的，所以不用转置
        sem[bg] = (0, 0, 0)
        sem[passive] = (85, 85, 85)
        sem[agent_ctrl] = (170, 170, 170)
        sem[agent] = (255, 255, 255)
        
        # 写入一帧
        writer.write(sem)

    writer.release()
    logger.info("Saved RGB mask video to %s", semantic_video_path)

    # 【修复4】函数结尾再次清理
    del role_masks_per_frame
    del writer
    gc.collect()
    torch.cuda.empty_cache()


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
        "--video_prompt_csv",
        default="prompts/metadata.csv",
        help="Path to video prompt file.",
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
    parser.add_argument(
        "--start_idx", default=0, type=int, help="the start idx to process videos dir"
    )
    parser.add_argument(
        "--process_len", default=100, type=int, help="the start idx to process videos dir"
    )
    return parser


def main():
    args = build_argparser().parse_args()
    input_path = Path(args.video)
    
    # 鲁棒性改进: 如果 iter_mp4_files 找不到路径或抛出异常，优雅处理
    try:
        video_paths = iter_mp4_files(input_path)
    except FileNotFoundError:
        logger.error(f"Input path not found: {input_path}")
        return
    except Exception as e:
        logger.error(f"Error finding video files: {e}")
        return

    if input_path.is_dir():
        logger.info("Found %d mp4 files under %s", len(video_paths), input_path)
    
    if not video_paths:
        logger.warning(f"No mp4 files found under: {input_path}")
        return

    # 记录失败的视频列表，方便最后统计
    failed_videos = []
    print("total video path nums:", len(video_paths))
    if args.start_idx+args.process_len >= len(video_paths):
        pbar = tqdm(video_paths[args.start_idx:], desc="Total Progress", unit="video")
    else:
        pbar = tqdm(video_paths[args.start_idx:args.start_idx+args.process_len], desc="Total Progress", unit="video")
    df = pd.read_csv(args.video_prompt_csv)
    df_lookup = df.set_index('video')

    def get_prompt_by_path(video_path):
        # 从完整路径提取文件名 (例如: /data/vids/abc.mp4 -> abc.mp4)
        filename = os.path.basename(video_path)
        
        # 使用 .loc 进行索引查找
        try:
            # 提取对应的 prompt
            return df_lookup.loc[filename, 'prompt']
        except KeyError:
            return "These is no Prompt."
    def und_part():
        model, processor = load_qwen_model(
            model_id=args.model_id, attn_implementation="flash_attention_2"
        )
        obj_id_to_role_list = {} 
        text_prompts_list = {}
        for vp in pbar:
        # for vp in video_paths:
            if check_rgb_exists(vp, args.output_dir):
                logger.info(f"{vp.name} is exists")
                print(f"{vp.name} is exists")
                continue
            try:
                logger.info("-" * 40)
                logger.info(f"Starting pipeline for: {vp.name}")
                obj_id_to_role, text_prompts = run_pipeline_und(
                    model, processor, None,
                    video_path=str(vp),
                    video_prompt=get_prompt_by_path(str(vp)),
                    system_prompt_path=args.system_prompt,
                    output_dir=args.output_dir,
                    model_id=args.model_id,
                    max_new_tokens=args.max_new_tokens,
                )
                obj_id_to_role_list[vp] = obj_id_to_role 
                text_prompts_list[vp] = text_prompts
                logger.info(f"Successfully processed: {vp.name}")
            except KeyboardInterrupt:
                # 允许用户通过 Ctrl+C 终止整个程序
                logger.info("Process interrupted by user.")
                break
            except Exception:
                # 捕获所有其他异常，打印堆栈并继续
                logger.error(f"Failed to process video: {vp}")
                logger.error(traceback.format_exc())
                failed_videos.append(str(vp))
                continue
        del model
        del processor
        torch.cuda.empty_cache()
        return obj_id_to_role_list, text_prompts_list

    obj_id_to_role_list, text_prompts_list = und_part()
    if args.start_idx+args.process_len >= len(video_paths):
        pbar = tqdm(video_paths[args.start_idx:], desc="Total Progress", unit="video")
    else:
        pbar = tqdm(video_paths[args.start_idx:args.start_idx+args.process_len], desc="Total Progress", unit="video")
    for vp in pbar:
    # for vp in video_paths:
        if check_rgb_exists(vp, args.output_dir) or vp not in obj_id_to_role_list:
            logger.info(f"{vp.name} is exists")
            print(f"{vp.name} is exists")
            continue
        try:
            obj_id_to_role = obj_id_to_role_list[vp]
            text_prompts = text_prompts_list[vp]
            logger.info("-" * 40)
            logger.info(f"Starting pipeline for: {vp.name}")
            run_pipeline_mask(
                None, None, None,
                video_path=str(vp),
                video_prompt=get_prompt_by_path(str(vp)),
                system_prompt_path=args.system_prompt,
                output_dir=args.output_dir,
                model_id=args.model_id,
                max_new_tokens=args.max_new_tokens,
                obj_id_to_role=obj_id_to_role,
                text_prompts=text_prompts,

            )
            logger.info(f"Successfully processed: {vp.name}")
            print(f"Successfully processed: {vp.name}")
        except KeyboardInterrupt:
            # 允许用户通过 Ctrl+C 终止整个程序
            logger.info("Process interrupted by user.")
            break
        except Exception:
            # 捕获所有其他异常，打印堆栈并继续
            logger.error(f"Failed to process video: {vp}")
            logger.error(traceback.format_exc())
            failed_videos.append(str(vp))
            continue
    # 最终报告
    logger.info("=" * 40)
    logger.info("Batch processing finished.")
    if failed_videos:
        logger.error(f"{len(failed_videos)} videos failed:")
        for fv in failed_videos:
            logger.error(f" - {fv}")
    else:
        logger.info("All videos processed successfully.")


if __name__ == "__main__":
    main()