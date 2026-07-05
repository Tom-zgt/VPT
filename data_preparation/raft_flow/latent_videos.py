import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Optional, Dict, Any
import inspect
from tqdm import tqdm
# from raft import RAFT
# from utils.utils import InputPadder
from diffusers import AutoencoderKLWan

def save_video_mp4(frames_uint8_thwc: np.ndarray, save_path: str, fps: int = 8):
    # frames: [T,H,W,3] RGB uint8
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    t, h, w, _ = frames_uint8_thwc.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for i in range(t):
        bgr = cv2.cvtColor(frames_uint8_thwc[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

@torch.no_grad()
def decode_and_save_examples(
    vae: AutoencoderKLWan,
    latent_dir: str,
    save_dir: str,
    num_examples: int = 3,
    fps: int = 8,
):
    os.makedirs(save_dir, exist_ok=True)
    pt_files = [os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith(".pt")]
    pt_files.sort()
    pt_files = pt_files[:num_examples]

    device = vae.device
    dtype = vae.dtype

    for pt in pt_files:
        lat = torch.load(pt, map_location="cpu")  # expect [1,C,T,H,W] or similar
        lat = lat.to(device=device, dtype=dtype)
        print("latent shape", lat.shape)
        print("latent min, max:", lat.min(), lat.max())
        mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype)
        std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype)

        # reshape mean/std to broadcast over [B,C,T,H,W] or [B,T,C,H,W] depending on your VAE
        # 你encode时 video 进 VAE 前做了 permute(0,2,1,3,4)，所以 latent 通常是 [B,C,T,H,W]
        if mean.ndim == 1:
            mean = mean.view(1, -1, 1, 1, 1)
            std = std.view(1, -1, 1, 1, 1)

        raw_lat = lat / (1.0 / std)  # 注意：你保存时 std = 1/latents_std
        raw_lat = raw_lat + mean
        print("raw latent shape", raw_lat.shape)
        print("raw latent min, max:", raw_lat.min(), raw_lat.max())
        # 更清晰的写法（推荐你用这个替换上面两行）：
        # saved_std = 1.0 / latents_std  (你代码就是这么定义的)
        # final = (raw - mean) * saved_std
        # => raw = final / saved_std + mean
        # raw_lat = lat / std + mean    # std 在你processor里就是 saved_std
        # raw_lat = lat / std + mean

        # decode
        decoded = vae.decode(raw_lat).sample  # 通常返回 [B,C,T,H,W]
        print("decoded shape", decoded.shape)
        print("decoded min, max:", decoded.min(), decoded.max())
        vid = decoded[0]  # [C,T,H,W] or [T,C,H,W] 取决于实现

        # 尝试把输出整理成 [T,H,W,3]
        if vid.ndim == 4 and vid.shape[0] == 3:
            # [3,T,H,W] -> [T,H,W,3]
            vid = vid.permute(1, 2, 3, 0)
        elif vid.ndim == 4 and vid.shape[1] == 3:
            # [T,3,H,W] -> [T,H,W,3]
            vid = vid.permute(0, 2, 3, 1)
        else:
            raise RuntimeError(f"Unexpected decoded shape: {decoded.shape}")

        vid = vid.float().clamp(-1, 1)
        vid = ((vid + 1.0) * 0.5 * 255.0).byte().cpu().numpy()

        out_name = os.path.splitext(os.path.basename(pt))[0] + "_decoded.mp4"
        for t in range(0, vid.shape[0], 1):
            rgb = vid[t]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"{os.path.splitext(os.path.basename(pt))[0]}_t{t:03d}.png"), bgr)
        save_video_mp4(vid, os.path.join(save_dir, out_name), fps=fps)
        print(f"[decode_check] saved: {out_name}, shape={vid.shape}")

class ProcessorMixin:
    def __init__(self) -> None:
        self._forward_parameter_names = inspect.signature(self.forward).parameters.keys()
        self.output_names: List[str] = None
        self.input_names: Dict[str, Any] = None

    def __call__(self, *args, **kwargs) -> Any:
        shallow_copy_kwargs = dict(kwargs.items())
        if self.input_names is not None:
            for k, v in self.input_names.items():
                if k in shallow_copy_kwargs:
                    shallow_copy_kwargs[v] = shallow_copy_kwargs.pop(k)
        acceptable_kwargs = {k: v for k, v in shallow_copy_kwargs.items() if k in self._forward_parameter_names}
        output = self.forward(*args, **acceptable_kwargs)
        if "__drop__" in output:
            output.pop("__drop__")
        return output

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("ProcessorMixin::forward method should be implemented by the subclass.")


class WanLatentEncodeProcessor(ProcessorMixin):
    def __init__(self, output_names: List[str]):
        super().__init__()
        self.output_names = output_names
        assert len(self.output_names) == 3

    def forward(
        self,
        vae: AutoencoderKLWan,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype
        if image is not None:
            video = image.unsqueeze(1)
        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"

        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        # print("video shape", video.shape)
        # print("video min max:", video.min(), video.max())
        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean).to(device, dtype=dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).to(device, dtype=dtype)

        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}


def flow_to_videojam_rgb(flow):
    h, w, _ = flow.shape
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    diag = np.sqrt(h**2 + w**2)
    sigma = 0.15
    rho = np.sqrt(u**2 + v**2)
    norm_factor = sigma * diag + 1e-6
    m = rho / norm_factor
    m = np.minimum(1.0, m)
    angle = np.arctan2(v, u)
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    angle_deg = (angle * 180 / np.pi) + 180
    hsv[:, :, 0] = angle_deg / 2.0
    hsv[:, :, 1] = m * 255.0
    hsv[:, :, 2] = 255.0
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr


def load_video_frames(video_path, device, target_frames=81, width=832, height=480):
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    num_frames = len(all_frames)
    if num_frames < 2:
        return None

    indices = torch.linspace(0, num_frames - 1, target_frames).long()

    frames_buffer = []
    for idx in indices:
        frame = all_frames[idx.item()]
        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()#.to(device)
        frames_buffer.append(frame_tensor)
    return torch.stack(frames_buffer, dim=0)


def compute_flow_batch(raft_model, frames_buffer, batch_size=16, device='cuda'):
    flows = []
    num_pairs = len(frames_buffer) - 1

    for start_idx in range(0, num_pairs, batch_size):
        end_idx = min(start_idx + batch_size, num_pairs)
        actual_batch_size = end_idx - start_idx

        image1_batch = torch.stack(frames_buffer[start_idx:end_idx], dim=0).to(device)  # [B, C, H, W]
        image2_batch = torch.stack(frames_buffer[start_idx + 1:end_idx + 1], dim=0).to(device)  # [B, C, H, W]

        padder = InputPadder(image1_batch.shape)
        image1_batch, image2_batch = padder.pad(image1_batch, image2_batch)

        with torch.no_grad():
            _, flow_up_batch = raft_model(image1_batch, image2_batch, iters=20, test_mode=True)

        flow_up_batch = flow_up_batch.cpu()
        for i in range(actual_batch_size):
            flow = flow_up_batch[i].permute(1, 2, 0).numpy()  # [H, W, 2]
            flows.append(flow)
    return flows


def process_single_video(video_path, output_dir, raft_model, vae_model, vae_processor, device):
    video_name = os.path.basename(video_path)
    file_stem = os.path.splitext(video_name)[0]
    save_path = os.path.join(output_dir, f"{file_stem}.pt")

    if os.path.exists(save_path):
        print(f"Skipping {video_name}, already exists.")
        return

    print(f"Processing: {video_name}")

    frames_buffer = load_video_frames(video_path, device=device, target_frames=81, width=832, height=480)
    if frames_buffer is None:
        print(f"Error: Could not load frames for {video_name}")
        return

    flow_rgb_list = frames_buffer #[]

    # with torch.no_grad():
    #     for i in range(len(frames_buffer) - 1):
    #         image1 = frames_buffer[i][None]  # add batch dim
    #         image2 = frames_buffer[i + 1][None]

    #         padder = InputPadder(image1.shape)
    #         image1, image2 = padder.pad(image1, image2)

    #         _, flow_up = raft_model(image1, image2, iters=20, test_mode=True)
    #         flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

    #         bgr_flow_img = flow_to_videojam_rgb(flo)
    #         rgb_flow_img = cv2.cvtColor(bgr_flow_img, cv2.COLOR_BGR2RGB)

    #         flow_rgb_list.append(rgb_flow_img)

    if len(flow_rgb_list) > 0:
        # first_frame = flow_rgb_list[0].copy()
        # flow_rgb_list.insert(0, first_frame)
        assert len(flow_rgb_list) == 81, f"Expected 81 flow frames, got {len(flow_rgb_list)}"
    # flow_rgb_list = 
    # print("frames shape", frames_buffer.shape)
    # print("frames min max", frames_buffer.min(), frames_buffer.max())
    frames = flow_rgb_list.to(device=device)  # float32
    frames = (frames / 255.0 - 0.5) * 2.0
    vae_input = frames.unsqueeze(0).to(dtype=vae_model.dtype)

    with torch.no_grad():
        outputs = vae_processor.forward(
            vae=vae_model,
            video=vae_input,
            compute_posterior=True
        )

    raw_latents = outputs["latents"]
    mean = outputs["latents_mean"]
    std = outputs["latents_std"]
    # print("raw latents shape", raw_latents.shape)
    # print("raw latents min max:", raw_latents.min(), raw_latents.max())
    # print("mean shape", mean.shape)
    # print("std shape", std.shape)
    if mean.ndim == 1:
        mean = mean.view(1, -1, 1, 1, 1)
        std = std.view(1, -1, 1, 1, 1)

    final_latents = (raw_latents.float() - mean.float()) * std.float()
    final_latents = final_latents.to(raw_latents.dtype)
    # decode_and_save_examples(
    #     vae=vae_model,
    #     latent_dir='./test_latents/',
    #     save_dir=os.path.join('./test_latents/', "_decode_check"),
    #     num_examples=3,
    #     fps=8,
    # )
    # assert 1 == 2
    torch.save(final_latents.cpu(), save_path)
    # print(f"Saved normalized latents to {save_path} | Shape: {final_latents.shape}")




from tqdm import tqdm

def worker_main(rank, world_size, args, video_files):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    print(f"[Rank {rank}] Using device {device}")

    # raft = RAFT(args)
    # state_dict = torch.load(args.raft_model, map_location='cpu', weights_only=True)
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     name = k[7:] if k.startswith('module.') else k
    #     new_state_dict[name] = v
    # raft.load_state_dict(new_state_dict)
    # raft.to(device)
    # raft.eval()

    vae = AutoencoderKLWan.from_pretrained(args.vae_checkpoint, subfolder="vae")
    vae = vae.to(device)
    if torch.cuda.is_bf16_supported():
        vae = vae.to(torch.bfloat16)
    else:
        vae = vae.to(torch.float16)
    vae.eval()
    # decode_and_save_examples(
    #     vae=vae,
    #     latent_dir=args.output_dir,
    #     save_dir=os.path.join('./test_latents/', "_decode_check"),
    #     num_examples=10,
    #     fps=8,
    # )
    # return
    processor = WanLatentEncodeProcessor(output_names=["latents", "latents_mean", "latents_std"])

    my_files = video_files[rank::world_size]
    print(f"[Rank {rank}] Assigned {len(my_files)} videos")
    iii = 0
    for video_path in tqdm(my_files, desc=f"[Rank {rank}] Processing videos"):
        try:
            process_single_video(video_path, args.output_dir, None, vae, processor, device)
        except Exception as e:
            print(f"[Rank {rank}] Error processing {video_path}: {e}")


def main():
    parser = argparse.ArgumentParser()
    # RAFT 参数
    parser.add_argument('--raft_model', default="models/raft-things.pth", help="restore RAFT checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')

    # 任务参数
    parser.add_argument('--video_folder', default="data/videos_data/videos", help="input video folder")
    parser.add_argument('--output_dir', default="data/latent_data/flow_videos", help="output folder for latents")

    # VAE 参数
    parser.add_argument('--vae_checkpoint', default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="vae checkpoint or repo id")

    # 性能参数
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help="number of GPUs to use")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [
        os.path.join(args.video_folder, f)
        for f in os.listdir(args.video_folder)
        if f.lower().endswith(video_extensions)
    ]
    video_files.sort()

    print(f"Total videos found: {len(video_files)}")
    print(f"Spawning {args.num_gpus} processes...")

    mp.spawn(
        worker_main,
        args=(args.num_gpus, args, video_files),
        nprocs=args.num_gpus,
        join=True
    )

    print("All processing done.")


if __name__ == '__main__':
    main()
