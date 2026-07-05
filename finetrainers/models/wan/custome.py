import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Sequence, Union, List, Optional, Tuple, Dict

import torch
import torch.nn as nn
from safetensors.torch import load_file
from diffusers import WanPipeline, WanImageToVideoPipeline
from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput


class CustomWanPipeline:
    def __init__(self, base_pipeline: Union[WanPipeline, WanImageToVideoPipeline]):
        self._base_pipeline = base_pipeline
        
    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(self._base_pipeline, name):
                return getattr(self._base_pipeline, name)
            else:
                raise AttributeError(f"'{type(self).__name__}' and '{type(self._base_pipeline).__name__}' object has no attribute '{name}'")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        base_pipeline = WanPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(base_pipeline)

    def to(self, device):
        self._base_pipeline.to(device)
        return self

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict, adapter_name="default", **kwargs):
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict = pretrained_model_name_or_path_or_dict
        else:
            weights_path = pretrained_model_name_or_path_or_dict
            if os.path.isdir(weights_path):
                weights_path = os.path.join(weights_path, "pytorch_lora_weights.safetensors")
            
            if not os.path.exists(weights_path):
                 print(f"Warning: Could not find local file at {weights_path}, delegating to standard loader.")
                 return self._base_pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict, **kwargs)
            
            state_dict = load_file(weights_path)

        custom_layer_keys = {}
        lora_keys = {}
        
        target_modules = ["patch_embedding", "proj_out", "time_embedder", "time_proj", "scale_shift_table", "cond_proj"]
        
        for k, v in state_dict.items():
            is_custom = any(m in k for m in target_modules) and "lora" not in k
            if is_custom:
                clean_key = k.replace("transformer.", "")
                custom_layer_keys[clean_key] = v
            else:
                lora_keys[k] = v

        if "patch_embedding.weight" in custom_layer_keys:
            new_weight = custom_layer_keys["patch_embedding.weight"]
            new_in_channels = new_weight.shape[1] 
            
            current_layer = self._base_pipeline.transformer.patch_embedding
            
            if new_in_channels != current_layer.in_channels:
                print(f"[CustomWanPipeline] Resizing patch_embedding: {current_layer.in_channels} -> {new_in_channels} channels")
                
                new_layer = nn.Conv3d(
                    in_channels=new_in_channels,
                    out_channels=current_layer.out_channels,
                    kernel_size=current_layer.kernel_size,
                    stride=current_layer.stride,
                    padding=current_layer.padding,
                ).to(dtype=self._base_pipeline.transformer.dtype, device=self._base_pipeline.device)
                
                self._base_pipeline.transformer.patch_embedding = new_layer

        if "proj_out.weight" in custom_layer_keys:
            new_weight = custom_layer_keys["proj_out.weight"]
            new_out_features = new_weight.shape[0]
            
            current_layer = self._base_pipeline.transformer.proj_out
            
            if isinstance(current_layer, nn.Linear) and new_out_features != current_layer.out_features:
                print(f"[CustomWanPipeline] Resizing proj_out: {current_layer.out_features} -> {new_out_features} features")
                
                new_layer = nn.Linear(
                    in_features=current_layer.in_features,
                    out_features=new_out_features,
                    bias=True if custom_layer_keys.get("proj_out.bias") is not None else False
                ).to(dtype=self._base_pipeline.transformer.dtype, device=self._base_pipeline.device)
                
                self._base_pipeline.transformer.proj_out = new_layer

        if custom_layer_keys:
            has_cond_proj = any("cond_proj" in k for k in custom_layer_keys)
            if has_cond_proj:
                from finetrainers.models.wan.base_specification import _maybe_enable_triple_discrete_timestep_conditioning
                print("[CustomWanPipeline] Detected cond_proj weights — enabling triple discrete timestep conditioning.")
                _maybe_enable_triple_discrete_timestep_conditioning(self._base_pipeline.transformer)
            print(f"[CustomWanPipeline] Loading full weights for: {list(custom_layer_keys.keys())}")
            self._base_pipeline.transformer.load_state_dict(custom_layer_keys, strict=False)

        if lora_keys:
            print(f"[CustomWanPipeline] Loading {len(lora_keys)} LoRA adapters...")
            self._base_pipeline.load_lora_weights(lora_keys, adapter_name=adapter_name, **kwargs)
        else:
            print("[CustomWanPipeline] No LoRA keys found in checkpoint.")
        
        return self


    def prepare_image_latents(
        self, image, last_image, batch_size, num_frames, height, width, generator, dtype, device
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError("`image` must be a torch.Tensor")
            
        video = image.unsqueeze(2) 
        if last_image is not None:
            last_image = last_image.unsqueeze(2)
            video = torch.cat([video, last_image], dim=2)
            
        if video.shape[2] < num_frames:
             padding = torch.zeros(
                 (video.shape[0], video.shape[1], num_frames - video.shape[2], video.shape[3], video.shape[4]),
                 dtype=video.dtype,
                 device=video.device,
             )
             video = torch.cat([video, padding], dim=2)
             
        video = video.permute(0, 2, 1, 3, 4) # [B, F, C, H, W]
        video = video.permute(0, 2, 1, 3, 4).contiguous() # [B, C, F, H, W]
        temporal_downsample = 4
        
        if last_image is None:
            first_frame, remaining_frames = video[:, :, :1], video[:, :, 1:]
            video_cond = torch.cat([first_frame, torch.zeros_like(remaining_frames)], dim=2)
        else:
            first_frame, remaining_frames, last_frame = video[:, :, :1], video[:, :, 1:-1], video[:, :, -1:]
            video_cond = torch.cat([first_frame, torch.zeros_like(remaining_frames), last_frame], dim=2)
        
        i2v_latents = self.vae.encode(video_cond).latent_dist.mode() 
        i2v_latents = i2v_latents.to(dtype=dtype)
        
        num_frames_latent = num_frames // temporal_downsample
        mask_temporal = i2v_latents.new_ones(num_frames_latent)
        if last_image is None:
            mask_temporal[1:] = 0
        else:
            mask_temporal[1:-1] = 0
            
        i2v_mask = mask_temporal.view(1, 1, -1, 1, 1).expand(
            i2v_latents.shape[0], 1, -1, i2v_latents.shape[3], i2v_latents.shape[4]
        )
        
        return i2v_latents, i2v_mask



    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float =1.8,
        flow_guidance_scale: float = 0.8,
        seg_guidance_scale: float = 0.8,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        
        image: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        
        dino_generator: Optional[torch.Generator] = None,
        flow_generator: Optional[torch.Generator] = None,
        seg_generator: Optional[torch.Generator] = None,
        enable_inner_guidance: bool = True,
        **kwargs,
    ):
        self._base_pipeline._guidance_scale = guidance_scale
        self._base_pipeline._attention_kwargs = attention_kwargs

        self.check_inputs(prompt, negative_prompt, height, width, prompt_embeds, negative_prompt_embeds)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device
        transformer_dtype = self.transformer.dtype

        config = self.transformer.config
        try:
            vae_out_ch = config.original_out_channels
            vae_in_ch = config.original_in_channels
        except AttributeError as e:
            raise AttributeError(
                "Transformer config is missing dual-stream attributes. "
                "Did you load the correct DualStreamTransformer3DModel? "
                f"Error: {e}"
            )

        # Compat: different implementations may name the flow stream channels differently.
        # - This repo's DualStreamTransformer3DModel registers `flow_in_channels` / `flow_out_channels`.
        # - Some Wan implementations may expose `mae_in_channels` / `mae_out_channels`.
        flow_in_ch = getattr(config, "mae_in_channels", None)
        seg_in_ch = getattr(config, "mae_in_channels", None)
        if flow_in_ch is None:
            flow_in_ch = getattr(config, "flow_in_channels", None)
        if flow_in_ch is None and hasattr(config, "in_channels"):
            flow_in_ch = (config.in_channels - vae_in_ch)//2 #zgt add //2
        if seg_in_ch is None:
            seg_in_ch = getattr(config, "seg_in_channels", None)
        if seg_in_ch is None and hasattr(config, "in_channels"):
            seg_in_ch = config.in_channels - vae_in_ch - flow_in_ch

        flow_out_ch = getattr(config, "mae_out_channels", None)
        if flow_out_ch is None:
            flow_out_ch = getattr(config, "flow_out_channels", None)
        if flow_out_ch is None and hasattr(config, "out_channels"):
            flow_out_ch = (config.out_channels - vae_out_ch)//2 #zgt add //2
        seg_out_ch = getattr(config, "mae_out_channels", None)
        if seg_out_ch is None:
            seg_out_ch = getattr(config, "seg_out_channels", None)
        if seg_out_ch is None and hasattr(config, "out_channels"):
            seg_out_ch = config.out_channels - vae_out_ch - flow_out_ch
        

        if flow_in_ch is None or flow_out_ch is None:
            raise AttributeError(
                "Transformer config is missing flow stream channel attributes. "
                "Expected one of (`flow_in_channels`/`flow_out_channels`) or (`mae_in_channels`/`mae_out_channels`)."
            )
        if seg_in_ch is None or seg_out_ch is None:
            raise AttributeError(
                "Transformer config is missing seg stream channel attributes. "
                "Expected one of (`seg_in_channels`/`seg_out_channels`) or (`mae_in_channels`/`mae_out_channels`)."
            )
        is_i2v = "image_dim" in self.transformer.config and self.transformer.config.image_dim is not None
        i2v_ch = 0
        if is_i2v:
            i2v_ch = vae_in_ch - vae_out_ch
            if i2v_ch <= 0:
                 raise ValueError("I2V model detected, but original_in_channels <= original_out_channels. "
                                  "Config is incorrect.")


        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # [Joint_Cond, Uncond, No_FLOW]
        # Prompt structure: [Pos, Neg, Pos]
        if enable_inner_guidance:
            prompt_embeds_input = torch.cat([
                prompt_embeds,           # Joint
                negative_prompt_embeds,  # Uncond
                prompt_embeds,            # No-FLOW
                prompt_embeds,             # No-SEG
            ])
        else:
            # Positive, Negative
            prompt_embeds_input = torch.cat([prompt_embeds, negative_prompt_embeds])

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        i2v_conditioning = None
        if is_i2v:
            if image is None:
                raise ValueError("`image` must be provided for I2V models.")
            i2v_latents, i2v_mask = self.prepare_image_latents(
                 image=image,
                 last_image=last_image,
                 batch_size=batch_size * num_videos_per_prompt,
                 num_frames=num_frames,
                 height=height,
                 width=width,
                 generator=generator,
                 dtype=transformer_dtype,
                 device=device
            )
            i2v_conditioning = torch.cat([i2v_latents, i2v_mask], dim=1)
        
        
        latents_vae = self.prepare_latents(
            batch_size * num_videos_per_prompt, vae_out_ch, height, width,
            num_frames, transformer_dtype, device, generator, None,
        )
        
        
        if flow_generator is None:
            flow_generator = torch.Generator(device=device)
            flow_generator.manual_seed(generator.initial_seed() + 2 if generator else 0)
        if seg_generator is None:
            seg_generator = torch.Generator(device=device)
            seg_generator.manual_seed(generator.initial_seed() + 3 if generator else 0)

        latents_flow = self.prepare_latents(
            batch_size * num_videos_per_prompt, flow_in_ch, height, width,
            num_frames, transformer_dtype, device, flow_generator, None,
        )

        latents_seg = self.prepare_latents(
            batch_size * num_videos_per_prompt, seg_in_ch, height, width,
            num_frames, transformer_dtype, device, seg_generator, None,
        )


        if is_i2v:
            latents = torch.cat([latents_vae, i2v_conditioning, latents_flow, latents_seg], dim=1)
        else:
            latents = torch.cat([latents_vae, latents_flow, latents_seg], dim=1)
        
        if isinstance(num_inference_steps, torch.Tensor):
            num_inference_steps = int(num_inference_steps.item())

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Inner-Guidance Logic
                # 1. Joint: [VAE, FLOW, SEG] + Text
                # 2. Uncond: [VAE, FLOW, SEG] + Neg
                # 3. No-FLOW: [VAE, 0, SEG] + Text
                # 4. No-SEG: [VAE, FLOW, 0] + Text
                current_latents = latents.to(device=device, dtype=transformer_dtype)
                
                if enable_inner_guidance:
                    # Joint & Uncond
                    batch_joint_uncond = torch.cat([current_latents, current_latents])
                    base_ch = vae_out_ch + (i2v_ch if is_i2v else 0)
                    flow_start = base_ch
                    seg_start = base_ch + flow_in_ch
                    
                    # 3. No-FLOW
                    latents_no_flow = current_latents.clone()
                    latents_no_flow[:, flow_start:seg_start, ...] = 0.0
                    # 4. No-SEG
                    latents_no_seg = current_latents.clone()
                    latents_no_seg[:, seg_start:, ...] = 0.0
                    # [Joint, Uncond, No-FLOW]
                    latent_model_input = torch.cat([
                        batch_joint_uncond,
                        latents_no_flow,
                        latents_no_seg,
                    ])

                else:
                    # CFG
                    base_ch = vae_out_ch + (i2v_ch if is_i2v else 0)
                    flow_start = base_ch
                    seg_start = base_ch + flow_in_ch
                    # current_latents[:, flow_start:, ...] = 0.0
                    latent_model_input = torch.cat([current_latents, current_latents])

                timestep = t.expand(latent_model_input.shape[0])

                # Forward Pass
                flow_timestep = timestep #torch.full_like(timestep, timesteps[0])
                seg_timestep = timestep #torch.full_like(timestep, timesteps[0])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    timestep_flow=flow_timestep,
                    timestep_seg=seg_timestep,
                    encoder_hidden_states=prompt_embeds_input,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                # Inner-Guidance Calculation
                if enable_inner_guidance:
                    # Joint, Uncond, No-FLOW, No-SEG
                    pred_joint, pred_uncond, pred_no_flow, pred_no_seg = noise_pred.chunk(4)
                    # Chain-CFG:
                    # Text Guidance = Joint - Uncond
                    text_guidance = pred_joint - pred_uncond
                    # FLOW Guidance = Joint - No_FLOW
                    flow_guidance = pred_joint - pred_no_flow
                    # SEG Guidance = Joint - No_SEG
                    seg_guidance = pred_joint - pred_no_seg
                    final_pred = pred_joint + (guidance_scale * text_guidance) + (flow_guidance_scale * flow_guidance) + (seg_guidance_scale * seg_guidance)
                else:
                    pred_cond, pred_uncond = noise_pred.chunk(2)
                    final_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # Scheduler Step & Output Handling
                if is_i2v:
                    pred_vae_final, pred_flow_final, pred_seg_final = torch.split(
                        final_pred,
                        [vae_out_ch, flow_out_ch, seg_out_ch],
                        dim=1
                    )
                    i2v_current = current_latents[:, vae_out_ch : vae_out_ch + i2v_ch, ...]
                    full_pred_for_step = torch.cat([
                        pred_vae_final,
                        i2v_current,
                        pred_flow_final,
                        pred_seg_final,
                    ], dim=1)
                    
                    latents = self.scheduler.step(full_pred_for_step, t, current_latents, return_dict=False)[0]
                else:
                    # if i < 25:
                    #     latents = self.scheduler.step(final_pred, t, current_latents, return_dict=False)[0]
                    # else:
                    latents = self.scheduler.step(final_pred, t, current_latents, return_dict=False)[0]
                    # latents[:,:flow_start] = self.scheduler.step(final_pred, t, current_latents, return_dict=False)[0][:,:flow_start]

                progress_bar.update()

        if not output_type == "latent":
            video_latents = latents[:, :vae_out_ch, :, :, :].to(self.vae.dtype)
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(video_latents.device, video_latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(video_latents.device, video_latents.dtype)
            video_latents = video_latents * latents_std + latents_mean
            video = self.vae.decode(video_latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
