import functools
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from torchvision.transforms import ToTensor
from .custome import CustomWanPipeline
import PIL.Image
import torch
import torch.nn as nn
import math
import numpy as np
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel as OriginalWanTransformer3DModel,
)
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.transformers.transformer_wan import WanImageEmbedding, WanTimeTextImageEmbedding
import imageio
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

import finetrainers.functional as FF
from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.processors import ProcessorMixin, T5Processor
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import get_non_null_items, safetensors_torch_save_function
import csv
logger = get_logger()


def _wan_eval_video_fps() -> float:
    raw = os.environ.get("WAN_EVAL_VIDEO_FPS", "16")
    try:
        return float(raw)
    except ValueError:
        return 16.0


def _wan_vbench_filename_stem(eval_style: str, ori_prompt: str) -> Optional[str]:
    """Match VBench official eval naming (see VBench-master vbench __init__.py)."""
    if eval_style == "vbench":
        stem = ori_prompt
    else:
        return None
    return stem.replace("/", "_").replace("\\", "_")


def _wan_fs_safe_stem(stem: str, name_suffix: str) -> str:
    """
    Keep a single path component (stem + name_suffix) within a byte budget so create/open
    does not fail on the local filesystem (commonly 255 bytes per component).

    name_suffix examples: '.mp4', '-0.mp4', '-12.mp4'.
    Override with WAN_MAX_VIDEO_BASENAME_BYTES (default 248 to leave margin under 255).
    """
    try:
        max_b = int(os.environ.get("WAN_MAX_VIDEO_BASENAME_BYTES", "248"))
    except ValueError:
        max_b = 248
    suf_b = len(name_suffix.encode("utf-8"))
    budget = max(16, max_b - suf_b)
    raw = stem.encode("utf-8")
    if len(raw) <= budget:
        return stem
    cut = raw[:budget].decode("utf-8", errors="ignore").rstrip(" .")
    if not cut:
        cut = "clip"
    logger.warning(
        "Filename stem truncated to fit %s-byte component limit (suffix=%r); original stem length was %d chars.",
        max_b,
        name_suffix,
        len(stem),
    )
    return cut


def _wan_video_record_csv_path(save_videos_dir: str) -> Optional[str]:
    """
    Where to append all_videos_record.csv. None = skip (WAN_DISABLE_VIDEO_RECORD_CSV).

    WAN_VIDEO_RECORD_CSV_PATH: absolute path to a .csv file, or a directory (we append
    all_videos_record.csv there).
    """
    flag = os.environ.get("WAN_DISABLE_VIDEO_RECORD_CSV", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return None
    raw = os.environ.get("WAN_VIDEO_RECORD_CSV_PATH", "").strip()
    if not raw:
        return os.path.join(save_videos_dir, "all_videos_record.csv")
    if raw.endswith(os.sep) or (os.path.isdir(raw) and not raw.endswith(".csv")):
        return os.path.join(raw.rstrip(os.sep), "all_videos_record.csv")
    return raw


def append_to_csv(csv_filepath, video_path, caption):
    """
    将单个视频的路径和提示词追加写入到 CSV 文件中。
    如果文件不存在，会自动创建并写入表头。
    """
    # 检查文件是否已经存在，用来判断是否需要先写表头
    file_exists = os.path.isfile(csv_filepath)
    
    # 使用 'a' (append) 模式打开文件，这样不会覆盖之前的数据
    with open(csv_filepath, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # 如果是第一次创建这个文件，先写一行表头
        if not file_exists:
            writer.writerow(['videopath', 'caption'])
        
        # 将当前生成的视频路径和 caption 追加到文件末尾
        writer.writerow([video_path, caption])


class WanTripleDiscreteTimeTextImageEmbedding(WanTimeTextImageEmbedding):
    """
    Extends diffusers `WanTimeTextImageEmbedding` to support two additional *discrete* timesteps for AdaLN
    time conditioning, without widening the main sinusoidal embedding path.
    Implementation detail:
    - Rebuild `time_embedder` as `TimestepEmbedding(..., cond_proj_dim=2 * freq_dim)`
    - Copy pretrained `linear_1/linear_2` weights
    - Zero-init `cond_proj` so that when `timestep_flow=timestep_seg=0`, the added residual is exactly zero
      (because `cond_proj([0,0,...]) == 0`).
    This avoids editing installed diffusers sources while preserving pretrained behavior at init.
    """
    def __init__(self, old: WanTimeTextImageEmbedding):
        dim = int(old.time_proj.in_features)
        time_freq_dim = int(old.timesteps_proj.num_channels)
        time_proj_dim = int(old.time_proj.out_features)
        text_embed_dim = int(old.text_embedder.linear_1.in_features)
        image_embed_dim = None
        pos_embed_seq_len = None
        if old.image_embedder is not None and isinstance(old.image_embedder, WanImageEmbedding):
            image_embed_dim = int(old.image_embedder.norm1.normalized_shape[0])
            pos = getattr(old.image_embedder, "pos_embed", None)
            pos_embed_seq_len = int(pos.shape[1]) if pos is not None else None
        super().__init__(
            dim=dim,
            time_freq_dim=time_freq_dim,
            time_proj_dim=time_proj_dim,
            text_embed_dim=text_embed_dim,
            image_embed_dim=image_embed_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.load_state_dict(old.state_dict(), strict=False)

        old_te: TimestepEmbedding = old.time_embedder
        new_te = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
            act_fn="silu",
            out_dim=None,
            post_act_fn=None,
            cond_proj_dim=2 * time_freq_dim,
            sample_proj_bias=True,
        ).to(device=old_te.linear_1.weight.device, dtype=old_te.linear_1.weight.dtype)
        with torch.no_grad():
            new_te.linear_1.weight.copy_(old_te.linear_1.weight)
            new_te.linear_1.bias.copy_(old_te.linear_1.bias)
            new_te.linear_2.weight.copy_(old_te.linear_2.weight)
            new_te.linear_2.bias.copy_(old_te.linear_2.bias)
            new_te.cond_proj.weight.zero_()
            if new_te.cond_proj.bias is not None:
                new_te.cond_proj.bias.zero_()
        self.time_embedder = new_te
        
    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
        timestep_flow: Optional[torch.Tensor] = None,
        timestep_seg: Optional[torch.Tensor] = None,
    ):
        inferred_ts_seq_len = timestep.shape[1] if timestep.ndim == 2 else None
        if timestep_seq_len is None:
            timestep_seq_len = inferred_ts_seq_len
        if timestep_flow is None:
            timestep_flow = torch.zeros_like(timestep)
        if timestep_seg is None:
            timestep_seg = torch.zeros_like(timestep)
        h_flow = self.timesteps_proj(timestep_flow)
        h_seg = self.timesteps_proj(timestep_seg)
        
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))
            if h_flow.ndim == 2:
                h_flow = h_flow.unflatten(0, (-1, timestep_seq_len))
            if h_seg.ndim == 2:
                h_seg = h_seg.unflatten(0, (-1, timestep_seq_len))
        cond = torch.cat([h_flow, h_seg], dim=-1)
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        if cond.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            cond = cond.to(time_embedder_dtype)

        temb = self.time_embedder(timestep, condition=cond).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image
def _maybe_enable_triple_discrete_timestep_conditioning(transformer: OriginalWanTransformer3DModel) -> None:
    old = transformer.condition_embedder
    if isinstance(old, WanTripleDiscreteTimeTextImageEmbedding):
        return
    if not isinstance(old, WanTimeTextImageEmbedding):
        raise TypeError(f"Unexpected condition_embedder type: {type(old)}")
    transformer.condition_embedder = WanTripleDiscreteTimeTextImageEmbedding(old)

        

class DualStreamTransformer3DModel(OriginalWanTransformer3DModel):

    def __init__(
        self,
        flow_in_channels: int = 0, 
        flow_out_channels: int = 0,
        seg_in_channels: int = 0, 
        seg_out_channels: int = 0,
        **kwargs,
    ):


        if "original_in_channels" in kwargs:
            original_in_channels = kwargs.pop("original_in_channels")
            original_out_channels = kwargs.pop("original_out_channels")
            new_in_channels = kwargs.pop("in_channels")
            new_out_channels = kwargs.pop("out_channels")

            super().__init__(
                in_channels=new_in_channels, 
                out_channels=new_out_channels, 
                **kwargs
            )

        else:
            original_in_channels = kwargs.pop("in_channels")
            original_out_channels = kwargs.pop("out_channels")

            new_in_channels = original_in_channels + flow_in_channels + seg_in_channels
            new_out_channels = original_out_channels + flow_out_channels + seg_out_channels

            super().__init__(
                in_channels=new_in_channels, 
                out_channels=new_out_channels, 
                **kwargs
            )

        config_dict = dict(self.config)

        config_dict['flow_in_channels'] = flow_in_channels
        config_dict['flow_out_channels'] = flow_out_channels

        config_dict['seg_in_channels'] = seg_in_channels
        config_dict['seg_out_channels'] = seg_out_channels

        config_dict['original_in_channels'] = original_in_channels
        config_dict['original_out_channels'] = original_out_channels

        self.register_to_config(**config_dict)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timestep_flow: Optional[torch.Tensor] = None,
        timestep_seg: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
        wan_logger = logging.get_logger(__name__)
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                wan_logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None
        if isinstance(self.condition_embedder, WanTripleDiscreteTimeTextImageEmbedding):
            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
                timestep_flow=timestep_flow,
                timestep_seg=timestep_seg,
            )
        else:
            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
            )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class FlowFeatureProcessor(ProcessorMixin):
    def __init__(self, output_names: List[str], flow_feature_root: str):
        super().__init__()
        assert len(output_names) == 1, "FlowFeatureProcessor should have exactly one output name."
        self.output_names = output_names
        self.flow_feature_root = flow_feature_root

    def forward(
            self,
            vae: AutoencoderKLWan,
            video_path: Optional[str] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if not video_path:
            logger.warning(f"video_path not in kwargs. Skipping Flow features.")
            return {self.output_names[0]: None}

        current_path = video_path[0] if isinstance(video_path, list) else video_path
        video_identifier = os.path.splitext(os.path.basename(current_path))[0]

        # flow_feature_path = os.path.join(self.flow_feature_root, f"{video_identifier}_dim8.npy")
        flow_feature_path = os.path.join(self.flow_feature_root, f"{video_identifier}.pt")

        if not os.path.exists(flow_feature_path):
            logger.warning(f"Flow feature file not found: {flow_feature_path}. Skipping.")
            return {self.output_names[0]: None}

        try:
            flow_latents = torch.load(flow_feature_path, map_location=device, weights_only=True)
            flow_latents = flow_latents.to(device=device, dtype=dtype)

        except Exception as e:
            raise RuntimeError(f"无法加载 Flow 特征文件: {flow_feature_path}") from e

        if flow_latents.ndim == 4:
            flow_latents = flow_latents.unsqueeze(0)

        reduce_dims = tuple(range(1, flow_latents.ndim))

        feature_mean = flow_latents.mean(dim=reduce_dims, keepdim=True)
        feature_std = flow_latents.std(dim=reduce_dims, keepdim=True)
        flow_latents = (flow_latents - feature_mean) / (feature_std + 1e-6)

        return {self.output_names[0]: flow_latents}
    
class SegFeatureProcessor(ProcessorMixin):
    def __init__(self, output_names: List[str], seg_feature_root: str):
        super().__init__()
        assert len(output_names) == 1, "SegFeatureProcessor should have exactly one output name."
        self.output_names = output_names
        self.seg_feature_root = seg_feature_root

    def forward(
            self,
            vae: AutoencoderKLWan,
            video_path: Optional[str] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype

        if not video_path:
            logger.warning(f"video_path not in kwargs. Skipping seg features.")
            return {self.output_names[0]: None}

        current_path = video_path[0] if isinstance(video_path, list) else video_path
        video_identifier = os.path.splitext(os.path.basename(current_path))[0]

        # flow_feature_path = os.path.join(self.flow_feature_root, f"{video_identifier}_dim8.npy")
        seg_feature_path = os.path.join(self.seg_feature_root, f"{video_identifier}_rgb.pt")

        if not os.path.exists(seg_feature_path):
            logger.warning(f"seg feature file not found: {seg_feature_path}. Skipping.")
            return {self.output_names[0]: None}

        try:
            seg_latents = torch.load(seg_feature_path, map_location=device, weights_only=True)
            seg_latents = seg_latents.to(device=device, dtype=dtype)

        except Exception as e:
            raise RuntimeError(f"无法加载 seg 特征文件: {seg_feature_path}") from e

        if seg_latents.ndim == 4:
            seg_latents = seg_latents.unsqueeze(0)

        reduce_dims = tuple(range(1, seg_latents.ndim))

        feature_mean = seg_latents.mean(dim=reduce_dims, keepdim=True)
        feature_std = seg_latents.std(dim=reduce_dims, keepdim=True)
        seg_latents = (seg_latents - feature_mean) / (feature_std + 1e-6)

        return {self.output_names[0]: seg_latents}


class WanLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.
    """

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
        if compute_posterior:
            latents = vae.encode(video).latent_dist.sample(generator=generator)
            latents = latents.to(dtype=dtype)
        else:
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)
        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)
        return {self.output_names[0]: latents, self.output_names[1]: latents_mean, self.output_names[2]: latents_std}


class WanImageConditioningLatentEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encode image/video into latents using the Wan VAE.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 4

    def forward(
            self,
            vae: AutoencoderKLWan,
            image: Optional[torch.Tensor] = None,
            video: Optional[torch.Tensor] = None,
            compute_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        device = vae.device
        dtype = vae.dtype
        if image is not None:
            video = image.unsqueeze(1)
        assert video.ndim == 5, f"Expected 5D tensor, got {video.ndim}D tensor"
        video = video.to(device=device, dtype=dtype)
        video = video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
        num_frames = video.size(2)
        if not self.use_last_frame:
            first_frame, remaining_frames = video[:, :, :1], video[:, :, 1:]
            video = torch.cat([first_frame, torch.zeros_like(remaining_frames)], dim=2)
        else:
            first_frame, remaining_frames, last_frame = video[:, :, :1], video[:, :, 1:-1], video[:, :, -1:]
            video = torch.cat([first_frame, torch.zeros_like(remaining_frames), last_frame], dim=2)
        if compute_posterior:
            latents = vae.encode(video).latent_dist.mode()
            latents = latents.to(dtype=dtype)
        else:
            moments = vae._encode(video)
            latents = moments.to(dtype=dtype)

        latents_mean = torch.tensor(vae.config.latents_mean)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std)
        temporal_downsample = 2 ** sum(vae.temperal_downsample) if getattr(self, "vae", None) else 4

        mask_temporal = latents.new_ones(num_frames // temporal_downsample)
        if not self.use_last_frame:
            mask_temporal[1:] = 0
        else:
            mask_temporal[1:-1] = 0

        mask = mask_temporal.view(1, 1, -1, 1, 1).expand(
            latents.shape[0], 1, -1, latents.shape[3], latents.shape[4]
        )

        return {
            self.output_names[0]: latents,
            self.output_names[1]: latents_mean,
            self.output_names[2]: latents_std,
            self.output_names[3]: mask,
        }


class WanImageEncodeProcessor(ProcessorMixin):
    r"""
    Processor to encoding image conditioning for Wan I2V training.
    """

    def __init__(self, output_names: List[str], *, use_last_frame: bool = False):
        super().__init__()
        self.output_names = output_names
        self.use_last_frame = use_last_frame
        assert len(self.output_names) == 1

    def forward(
            self,
            image_encoder: CLIPVisionModel,
            image_processor: CLIPImageProcessor,
            image: Optional[torch.Tensor] = None,
            video: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = image_encoder.device
        dtype = image_encoder.dtype
        last_image = None
        image = image if video is None else video[:, 0]  # [B, F, C, H, W] -> [B, C, H, W] (take first frame)
        image = FF.normalize(image, min=0.0, max=1.0, dim=1)
        assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor"
        if self.use_last_frame:
            last_image = image if video is None else video[:, -1]
            last_image = FF.normalize(last_image, min=0.0, max=1.0, dim=1)
            image = torch.stack([image, last_image], dim=0)
        image = image_processor(images=image.float(), do_rescale=False, do_convert_rgb=False, return_tensors="pt")
        image = image.to(device=device, dtype=dtype)
        image_embeds = image_encoder(**image, output_hidden_states=True)
        image_embeds = image_embeds.hidden_states[-2]
        return {self.output_names[0]: image_embeds}


class WanModelSpecification(ModelSpecification):
    def __init__(
            self,
            pretrained_model_name_or_path: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            save_videos_dir: Optional[str] = None,
            val_num_per_prompt: int = 1,
            val_batch_size: int = 1,
            validation_guidance_scale: float = 1.8,
            validation_flow_guidance_scale: float = 0.8,
            validation_seg_guidance_scale: float = 0.8,
            validation_enable_inner_guidance: bool = True,
            tokenizer_id: Optional[str] = None,
            text_encoder_id: Optional[str] = None,
            transformer_id: Optional[str] = None,
            vae_id: Optional[str] = None,
            text_encoder_dtype: torch.dtype = torch.bfloat16,
            transformer_dtype: torch.dtype = torch.bfloat16,
            vae_dtype: torch.dtype = torch.bfloat16,
            revision: Optional[str] = None,
            cache_dir: Optional[str] = None,
            condition_model_processors: List[ProcessorMixin] = None,
            latent_model_processors: List[ProcessorMixin] = None,

            flow_feature_root: str = "data/latent_data/flow_videos",

            flow_in_channels: int = 16,
            flow_out_channels: int = 16,
            seg_feature_root: str = "data/latent_data/seg_videos",
            seg_in_channels: int = 16,
            seg_out_channels: int = 16,
            **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )
        self.save_videos_dir = save_videos_dir
        self.val_num_per_prompt = val_num_per_prompt
        self.val_batch_size = val_batch_size
        self.validation_guidance_scale = validation_guidance_scale
        self.validation_flow_guidance_scale = validation_flow_guidance_scale
        self.validation_seg_guidance_scale = validation_seg_guidance_scale
        self.validation_enable_inner_guidance = validation_enable_inner_guidance

        self.flow_in_channels = flow_in_channels
        self.flow_out_channels = flow_out_channels
        self.seg_in_channels = seg_in_channels
        self.seg_out_channels = seg_out_channels



        use_last_frame = self.transformer_config.get("pos_embed_seq_len", None) is not None

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "__drop__"])]

        if latent_model_processors is None:
            latent_model_processors = [
                WanLatentEncodeProcessor(["latents", "latents_mean", "latents_std"]),
                FlowFeatureProcessor(
                    output_names=["flow_latents"],
                    flow_feature_root=flow_feature_root
                ),
                SegFeatureProcessor(
                    output_names=["seg_latents"],
                    seg_feature_root=seg_feature_root
                ),
            ]

        if self.transformer_config.get("image_dim", None) is not None:
            latent_model_processors.append(
                WanImageConditioningLatentEncodeProcessor(
                    ["latent_condition", "__drop__", "__drop__", "latent_condition_mask"],
                    use_last_frame=use_last_frame,
                )
            )
            latent_model_processors.append(
                WanImageEncodeProcessor(["encoder_hidden_states_image"], use_last_frame=use_last_frame)
            )

        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors

    @property
    def _resolution_dim_keys(self):
        return {"latents": (2, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}
        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )
        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )
        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}
        if self.vae_id is not None:
            vae = AutoencoderKLWan.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLWan.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )
        models = {"vae": vae}
        if self.transformer_config.get("image_dim", None) is not None:
            image_encoder = CLIPVisionModel.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_encoder", torch_dtype=torch.bfloat16
            )
            image_processor = CLIPImageProcessor.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="image_processor"
            )
            models["image_encoder"] = image_encoder
            models["image_processor"] = image_processor
        return models

    def load_diffusion_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        transformer_path_or_id = self.transformer_id or os.path.join(self.pretrained_model_name_or_path, "transformer")

        original_config = DualStreamTransformer3DModel.load_config(
            transformer_path_or_id, **common_kwargs
        )
        new_config = original_config.copy()

        new_config['flow_in_channels'] = self.flow_in_channels
        new_config['flow_out_channels'] = self.flow_out_channels
        new_config['seg_in_channels'] = self.seg_in_channels
        new_config['seg_out_channels'] = self.seg_out_channels



        transformer = DualStreamTransformer3DModel(**new_config)

        original_transformer = OriginalWanTransformer3DModel.from_pretrained(
            transformer_path_or_id, torch_dtype=self.transformer_dtype, **common_kwargs
        )
        original_sd = original_transformer.state_dict()
        mismatched_keys = ["patch_embedding.weight", "patch_embedding.bias", "proj_out.weight", "proj_out.bias"]
        copied_weights = {}

        for k in mismatched_keys:
            if k in original_sd:
                copied_weights[k] = original_sd.pop(k)
            else:
                raise KeyError(f"Key {k} not found in original checkpoint.")

        transformer.load_state_dict(original_sd, strict=False)
        logger.info("Backbone weights loaded successfully.")

        with torch.no_grad():
            old_in_w = copied_weights["patch_embedding.weight"]
            current_in_c = old_in_w.shape[1]

            transformer.patch_embedding.weight.data[:, :current_in_c, ...] = old_in_w
            transformer.patch_embedding.weight.data[:, current_in_c:, ...] = 0.0

            transformer.patch_embedding.bias.data = copied_weights["patch_embedding.bias"]

            old_out_w = copied_weights["proj_out.weight"]
            old_out_b = copied_weights["proj_out.bias"]

            patch_size = transformer.config.patch_size
            patch_vol = math.prod(patch_size)
            hidden_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim

            old_ch = old_out_w.shape[0] // patch_vol
            new_ch = transformer.config.out_channels

            old_out_w_reshaped = old_out_w.view(patch_vol, old_ch, hidden_dim)
            new_out_w_reshaped = torch.zeros(patch_vol, new_ch, hidden_dim,
                                             dtype=old_out_w.dtype, device=old_out_w.device)

            new_out_w_reshaped[:, :old_ch, :] = old_out_w_reshaped
            transformer.proj_out.weight.data = new_out_w_reshaped.reshape(patch_vol * new_ch, hidden_dim)

            old_out_b_reshaped = old_out_b.view(patch_vol, old_ch)
            new_out_b_reshaped = torch.zeros(patch_vol, new_ch, dtype=old_out_b.dtype, device=old_out_b.device)
            new_out_b_reshaped[:, :old_ch] = old_out_b_reshaped
            transformer.proj_out.bias.data = new_out_b_reshaped.reshape(patch_vol * new_ch)

        logger.info("Weight copy and zero-initialization complete for 4 streams.")

        del original_transformer
        del original_sd
        del copied_weights

        transformer.to(self.transformer_dtype)

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
            shift=5.0
        )

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
            self,
            tokenizer: Optional[AutoTokenizer] = None,
            text_encoder: Optional[UMT5EncoderModel] = None,
            transformer: Optional[DualStreamTransformer3DModel] = None,
            vae: Optional[AutoencoderKLWan] = None,
            scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
            image_encoder: Optional[CLIPVisionModel] = None,
            image_processor: Optional[CLIPImageProcessor] = None,
            enable_slicing: bool = False,
            enable_tiling: bool = False,
            enable_model_cpu_offload: bool = False,
            training: bool = False,
            **kwargs,
    ) -> Union[WanPipeline, WanImageToVideoPipeline]:

        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
        }

        is_i2v = self.transformer_config.get("image_dim", None) is not None

        if is_i2v:

            required_i2v = ["tokenizer", "text_encoder", "transformer", "vae", "scheduler", "image_encoder",
                            "image_processor"]
            missing_comps = [name for name in required_i2v if components.get(name) is None]
            if missing_comps:
                raise ValueError(f"WanImageToVideoPipeline 缺失必要组件: {missing_comps}")

            pipe = WanImageToVideoPipeline(
                tokenizer=components["tokenizer"],
                text_encoder=components["text_encoder"],
                transformer=components["transformer"],
                vae=components["vae"],
                scheduler=components["scheduler"],
                image_encoder=components["image_encoder"],
                image_processor=components["image_processor"],
            )
        else:

            required_t2v = ["tokenizer", "text_encoder", "transformer", "vae", "scheduler"]
            missing_comps = [name for name in required_t2v if components.get(name) is None]
            if missing_comps:
                raise ValueError(f"WanPipeline 缺失必要组件: {missing_comps}")

            pipe = WanPipeline(
                tokenizer=components["tokenizer"],
                text_encoder=components["text_encoder"],
                transformer=components["transformer"],
                vae=components["vae"],
                scheduler=components["scheduler"],
            )

        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)
        if not training:
            pipe.transformer.to(self.transformer_dtype)
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
            self,
            tokenizer: AutoTokenizer,
            text_encoder: UMT5EncoderModel,
            caption: str,
            max_sequence_length: int = 512,
            **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        return conditions

    @torch.no_grad()
    def prepare_latents(
            self,
            vae: AutoencoderKLWan,
            image_encoder: Optional[CLIPVisionModel] = None,
            image_processor: Optional[CLIPImageProcessor] = None,
            image: Optional[torch.Tensor] = None,
            video: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            compute_posterior: bool = True,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        conditions = {
            "vae": vae,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
            "image": image,
            "video": video,
            "generator": generator,
            "compute_posterior": False,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        processed_conditions = super().prepare_latents(**conditions)
        final_conditions = {k: v for k, v in processed_conditions.items() if k not in input_keys}
        return final_conditions

    def forward(
            self,
            transformer: DualStreamTransformer3DModel,
            condition_model_conditions: Dict[str, torch.Tensor],
            latent_model_conditions: Dict[str, torch.Tensor],
            sigmas: torch.Tensor,
            sigmas_flow: Optional[torch.Tensor] = None,
            sigmas_seg: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            compute_posterior: bool = True,
            **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        compute_posterior = False
        latent_condition = latent_condition_mask = None

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            # I2V specific
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)
        else:
            latents = latent_model_conditions.pop("latents")
            latents_mean = latent_model_conditions.pop("latents_mean")
            latents_std = latent_model_conditions.pop("latents_std")
            latent_condition = latent_model_conditions.pop("latent_condition", None)
            latent_condition_mask = latent_model_conditions.pop("latent_condition_mask", None)

            mu, logvar = torch.chunk(latents, 2, dim=1)
            mu = self._normalize_latents(mu, latents_mean, latents_std)
            logvar = self._normalize_latents(logvar, latents_mean, latents_std)
            latents = torch.cat([mu, logvar], dim=1)

            posterior = DiagonalGaussianDistribution(latents)
            latents = posterior.sample(generator=generator)

            # I2V Conditioning Normalization
            if latent_condition is not None:
                if latent_condition.shape[1] == (latents.shape[1] * 2):
                    mu, logvar = torch.chunk(latent_condition, 2, dim=1)
                    mu = self._normalize_latents(mu, latents_mean, latents_std)
                    logvar = self._normalize_latents(logvar, latents_mean, latents_std)
                    latent_condition = torch.cat([mu, logvar], dim=1)
                    posterior = DiagonalGaussianDistribution(latent_condition)
                    latent_condition = posterior.mode()
                elif latent_condition.shape[1] != latents.shape[1]:
                    raise ValueError("latent_condition has unexpected channel dimension.")
            del posterior
        # print("latents.shape:", latents.shape)
        flow_latents = latent_model_conditions.pop("flow_latents", None)
        seg_latents = latent_model_conditions.pop("seg_latents", None)

        # print(seg_latents)
        if seg_latents is not None:
            seg_latents = seg_latents[:,:,:latents.shape[2],:latents.shape[3],:latents.shape[4]]
        else:
            seg_latents = torch.zeros_like(latents)
        if flow_latents is not None:
            flow_latents = flow_latents[:,:,:latents.shape[2],:latents.shape[3],:latents.shape[4]]
        else:
            flow_latents = torch.zeros_like(latents)
        noise_vae = torch.zeros_like(latents).normal_(generator=generator)
        noise_flow = torch.zeros_like(flow_latents).normal_(generator=generator)
        noise_seg = torch.zeros_like(seg_latents).normal_(generator=generator)

        sigmas_flow = sigmas if sigmas_flow is None else sigmas_flow
        sigmas_seg = sigmas if sigmas_seg is None else sigmas_seg

        vae_noisy_latents = FF.flow_match_xt(latents, noise_vae, sigmas)
        flow_noisy_latents = FF.flow_match_xt(flow_latents, noise_flow, sigmas_flow)
        seg_noisy_latents = FF.flow_match_xt(seg_latents, noise_seg, sigmas_seg)

        timesteps = (sigmas.flatten() * 1000.0).long()
        timesteps_flow = None
        timesteps_seg = None
        if getattr(self, "multi_timestep_mode", "none") == "triple_discrete":
            timesteps_flow = (sigmas_flow.flatten() * 1000.0).long()
            timesteps_seg = (sigmas_seg.flatten() * 1000.0).long()

        if self.transformer_config.get("image_dim", None) is not None:
            if latent_condition is None or latent_condition_mask is None:
                raise ValueError("I2V model requires latent_condition and latent_condition_mask.")
            i2v_latents = torch.cat([latent_condition, latent_condition_mask], dim=1)
            vae_i2v_noisy_latents = torch.cat([vae_noisy_latents, i2v_latents], dim=1)
        else:
            vae_i2v_noisy_latents = vae_noisy_latents

        noisy_latents = torch.cat([vae_i2v_noisy_latents, flow_noisy_latents, seg_noisy_latents],
                                  dim=1)

        latent_model_conditions["hidden_states"] = noisy_latents.to(latents.dtype)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            extra_ts = {}
            if timesteps_flow is not None and timesteps_seg is not None:
                extra_ts.update({"timestep_flow": timesteps_flow, "timestep_seg": timesteps_seg})
            pred = transformer(
                **latent_model_conditions,
                **condition_model_conditions,
                timestep=timesteps,
                **extra_ts,
                return_dict=False,
            )[0]

        vae_out_channels = transformer.config.original_out_channels
        flow_out_channels = transformer.config.flow_out_channels
        seg_out_channels = transformer.config.seg_out_channels

        pred_vae,  pred_flow, pred_seg = torch.split(
            pred,
            [vae_out_channels, flow_out_channels, seg_out_channels],
            dim=1
        )

        target_vae = FF.flow_match_target(noise_vae, latents)
        target_flow = FF.flow_match_target(noise_flow, flow_latents)
        target_seg = FF.flow_match_target(noise_seg, seg_latents)

        return pred_vae, target_vae, pred_flow, target_flow, pred_seg, target_seg, sigmas

    def validation(
            self,
            pipeline: Union[WanPipeline, WanImageToVideoPipeline],
            prompt: str,
            ori_prompt: str,
            image: Optional[PIL.Image.Image] = None,
            last_image: Optional[PIL.Image.Image] = None,
            video: Optional[List[PIL.Image.Image]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_frames: Optional[int] = None,
            num_inference_steps: int = 50,
            generator: Optional[torch.Generator] = None,
            **kwargs,
    ) -> List[ArtifactType]:
        def _batch_item(x: Union[str, List[str], Tuple[str, ...]]) -> str:
            if isinstance(x, (list, tuple)):
                return x[0]
            return x

        per_row_val = kwargs.pop("val_num_per_prompt", None)
        try:
            if per_row_val is None:
                effective_val_num = int(self.val_num_per_prompt)
            else:
                effective_val_num = max(1, min(int(per_row_val), 128))
        except (TypeError, ValueError):
            effective_val_num = int(self.val_num_per_prompt)

        def _pop_float(key: str, default: float) -> float:
            v = kwargs.pop(key, None)
            if v is None:
                return float(default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        def _pop_bool(key: str, default: bool) -> bool:
            v = kwargs.pop(key, None)
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)

        guidance_scale = _pop_float("guidance_scale", self.validation_guidance_scale)
        flow_guidance_scale = _pop_float("flow_guidance_scale", self.validation_flow_guidance_scale)
        seg_guidance_scale = _pop_float("seg_guidance_scale", self.validation_seg_guidance_scale)
        enable_inner_guidance = _pop_bool("enable_inner_guidance", self.validation_enable_inner_guidance)

        # Match ``CustomWanPipeline.__call__`` defaults when the validation row omits geometry.
        if height is None:
            height = 480
        if width is None:
            width = 832
        if num_frames is None:
            num_frames = 81

        ori_s = _batch_item(ori_prompt)
        eval_style = os.environ.get("WAN_EVAL_STYLE", "")
        use_videorepa_names = eval_style in ("videophy", "videophy2")
        vbench_stem_raw = _wan_vbench_filename_stem(eval_style, ori_s)
        use_vbench_names = vbench_stem_raw is not None
        # Reserve bytes for ``-{index}.mp4`` so multi-sample (VBench) stays valid.
        vbench_stem = (
            _wan_fs_safe_stem(vbench_stem_raw, "-99.mp4")
            if use_vbench_names
            else None
        )
        eval_fps = _wan_eval_video_fps()

        if self.save_videos_dir and os.environ.get("WAN_SKIP_EXISTING_VIDEO", "") == "1":
            if use_videorepa_names and effective_val_num == 1:
                skip_stem = _wan_fs_safe_stem("_".join(ori_s.rstrip(".").split()), ".mp4")
                skip_path = os.path.join(self.save_videos_dir, skip_stem + ".mp4")
                if skip_path and os.path.isfile(skip_path):
                    logger.info("Skip validation (existing output): %s", skip_path)
                    return []
            elif use_vbench_names:
                all_exist = all(
                    os.path.isfile(os.path.join(self.save_videos_dir, f"{vbench_stem}-{i}.mp4"))
                    for i in range(effective_val_num)
                )
                if all_exist:
                    logger.info(
                        "Skip validation (vbench outputs exist 0..%s): %s",
                        effective_val_num - 1,
                        self.save_videos_dir,
                    )
                    return []

        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        prompt_s = _batch_item(prompt)
        generation_kwargs = {
            "prompt": prompt_s if effective_val_num > 1 else prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
            "guidance_scale": guidance_scale,
            "flow_guidance_scale": flow_guidance_scale,
            "seg_guidance_scale": seg_guidance_scale,
            "enable_inner_guidance": enable_inner_guidance,
        }
        device = pipeline.device
        pipeline = CustomWanPipeline(pipeline)
        pipeline = pipeline.to(device)
        if self.transformer_config.get("image_dim", None) is not None:
            if image is None and video is None:
                raise ValueError("Either image or video must be provided for Wan I2V validation.")
            image = image if image is not None else video[0]
            image_tensor = ToTensor()(image).unsqueeze(0).to(dtype=torch.float32, device=pipeline.device)
            generation_kwargs["image"] = image_tensor
        if self.transformer_config.get("pos_embed_seq_len", None) is not None:
            last_image = last_image if last_image is not None else image if video is None else video[-1]
            last_image_tensor = ToTensor()(last_image).unsqueeze(0).to(dtype=torch.float32, device=pipeline.device)
            generation_kwargs["last_image"] = last_image_tensor
        generation_kwargs = get_non_null_items(generation_kwargs)

        # Benchmark JSON may request e.g. 25 videos per prompt; run forwards in chunks to limit VRAM.
        try:
            cap_raw = int(os.environ.get("WAN_VALIDATION_MAX_VIDEOS_PER_FORWARD", "5"))
        except ValueError:
            cap_raw = 5
        max_videos_per_forward = max(1, min(cap_raw, 5))
        if effective_val_num > max_videos_per_forward:
            logger.info(
                "Wan validation: generating %s videos in chunks of at most %s (OOM-safe).",
                effective_val_num,
                max_videos_per_forward,
            )

        def _save_one_multi_index(video, index: int) -> None:
            video_name = ""
            os.makedirs(self.save_videos_dir, exist_ok=True)
            try:
                max_seqlen = 240
                log_prompt = prompt_s
                if len(log_prompt) > max_seqlen:
                    log_prompt = log_prompt[:max_seqlen]
                if use_vbench_names:
                    save_stem = vbench_stem
                else:
                    save_stem = _wan_fs_safe_stem(log_prompt, f"-{index}.mp4")
                video_name = f"{self.save_videos_dir}/{save_stem}-{index}.mp4"
                imageio.mimsave(video_name, video, fps=eval_fps, quality=10)
                print(f"✨ 视频已保存到本地: {video_name}")
            except Exception:
                print(f"can not save {video_name}")

        video = None
        if effective_val_num > 1:
            for offset in range(0, effective_val_num, max_videos_per_forward):
                chunk_n = min(max_videos_per_forward, effective_val_num - offset)
                kw = get_non_null_items({**generation_kwargs, "num_videos_per_prompt": chunk_n})
                chunk_frames = pipeline(**kw).frames
                for local_i, vid in enumerate(chunk_frames):
                    global_index = offset + local_i
                    _save_one_multi_index(vid, global_index)
                    video = vid
        else:
            kw_one = get_non_null_items({**generation_kwargs, "num_videos_per_prompt": 1})
            videos = pipeline(**kw_one).frames
            video = videos[0]
            # --- 新增：保存 PIL Image 列表到文件夹 ---
            # 建议增加时间戳或唯一 ID 文件夹，防止多次运行被覆盖
            os.makedirs(self.save_videos_dir, exist_ok=True)
            if use_videorepa_names:
                # VideoREPA inference/generate.py: f"{'_'.join(caption.rstrip('.').split(' '))}.mp4" (short caption)
                file_stem = _wan_fs_safe_stem("_".join(ori_s.rstrip(".").split()), ".mp4")
                csv_fname = f"{file_stem}.mp4"
                video_name = f"{self.save_videos_dir}/{file_stem}.mp4"
            elif use_vbench_names:
                # VBench: ``{prompt_en}-{i}.mp4`` (VBench-master).
                file_stem = vbench_stem
                csv_fname = f"{file_stem}-0.mp4"
                video_name = f"{self.save_videos_dir}/{csv_fname}"
            else:
                file_stem = _wan_fs_safe_stem(
                    f"{time.strftime('%Y%m%d-%H%M%S')}_{ori_s.replace(' ', '_')}",
                    ".mp4",
                )
                csv_fname = f"{file_stem}.mp4"
                video_name = f"{self.save_videos_dir}/{file_stem}.mp4"
            try:
                imageio.mimsave(video_name, video, fps=eval_fps, quality=10)
            except Exception:
                print(f"can not save {video_name}")
                pass

            print(f"✨ 视频已保存到本地: {video_name}")
            csv_target = _wan_video_record_csv_path(self.save_videos_dir)
            if csv_target:
                try:
                    csv_dir = os.path.dirname(os.path.abspath(csv_target))
                    if csv_dir:
                        os.makedirs(csv_dir, exist_ok=True)
                    append_to_csv(csv_target, csv_fname, ori_s)
                except OSError as exc:
                    logger.warning(
                        "Skip appending video record CSV (%s): %s. "
                        "Set WAN_DISABLE_VIDEO_RECORD_CSV=1 or WAN_VIDEO_RECORD_CSV_PATH to a local file.",
                        csv_target,
                        exc,
                    )
            
            
            # if isinstance(video, list):
            #     for i, img in enumerate(video):
            #         # img 是 PIL.Image 对象
            #         img_path = os.path.join(save_img_dir, f"frame_{i:04d}.png")
            #         img.save(img_path)
                
            #     print(f"✨ 视频帧已安全备份至目录: {save_img_dir} (共 {len(video)} 帧)")
        return [VideoArtifact(value=video)]

    def _save_lora_weights(
            self,
            directory: str,
            transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
            scheduler: Optional[SchedulerType] = None,
            metadata: Optional[Dict[str, str]] = None,
            *args,
            **kwargs,
    ) -> None:
        pipeline_cls = (
            WanImageToVideoPipeline if self.transformer_config.get("image_dim", None) is not None else WanPipeline
        )
        if transformer_state_dict is None:
            logger.error("[DEBUG] transformer_state_dict is None when calling _save_lora_weights()!")

        if transformer_state_dict is not None:
            pipeline_cls.save_lora_weights(
                directory,
                transformer_lora_layers=transformer_state_dict,
                save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                safe_serialization=True,
            )
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
            self,
            directory: str,
            transformer: DualStreamTransformer3DModel,
            transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
            scheduler: Optional[SchedulerType] = None,
    ) -> None:
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = DualStreamTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _normalize_latents(
            latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents
