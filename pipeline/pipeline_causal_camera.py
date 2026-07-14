from typing import List, Optional
import torch
import tqdm

from utils.wan_wrapper import WanDiffusionCameraWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalCameraInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None,
                 model_config_path=None, text_encoder_path=None, tokenizer_path=None, vae_path=None,
                 **kwargs):
        super().__init__()
        model_kwargs = getattr(args, "model_kwargs", {})
        model_name = model_kwargs["model_name"]
        model_root_path = model_kwargs["model_root_path"]

        self.generator = WanDiffusionCameraWrapper(
            **model_kwargs, is_causal=True, model_config_path=model_config_path,
            **kwargs) if generator is None else generator
        self.text_encoder = WanTextEncoder(
            model_name=model_name, model_root_path=model_root_path,
            text_encoder_path=text_encoder_path, tokenizer_path=tokenizer_path,
        ) if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper(
            model_root_path=model_root_path, vae_path=vae_path,
        ) if vae is None else vae

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880
        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        y: torch.Tensor,
        y_camera: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        chunk_callback=None,
        chunk_scorer=None,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size, noise.dtype, noise.device, num_frames)
            self._initialize_crossattn_cache(batch_size, noise.dtype, noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        # Cache initial latent frames
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    y=y[:, :1] if y is not None else None,
                    y_camera=y_camera if isinstance(y_camera, dict) else y_camera[:, :1],
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                y_latents = y[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                y_camera_latents = y_camera if isinstance(y_camera, dict) else y_camera[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    y=y_latents, y_camera=y_camera_latents,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        first_frame_mask = torch.zeros_like(noise)
        first_frame_mask[:, 1:] = 1

        for i, current_num_frames in tqdm.tqdm(enumerate(all_num_frames)):
            if profile:
                block_start.record()

            noisy_input = noise[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input
            y_latents = y[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames] if y is not None else None

            if isinstance(y_camera, dict):
                start_index = (current_start_frame - num_input_frames) * self.frame_seq_length
                end_index = start_index + self.frame_seq_length * current_num_frames
                y_camera_latents = {
                    'viewmats': y_camera['viewmats'][:, start_index:end_index],
                    'K': y_camera['K'][:, start_index:end_index],
                }
            else:
                y_camera_latents = y_camera[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            first_frame_mask_block = first_frame_mask[:, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Spatial denoising loop (as a closure so best-of-N re-roll can re-run it with
            # fresh noise; verbatim logic — byte-identical RNG path when chunk_scorer is None).
            # KV note: candidate 0's first step appends+advances cache pointers; every later
            # generator call at this position (more steps, more candidates, context rerun) is an
            # in-place recompute, so re-rolls never corrupt the cache bookkeeping.
            def _denoise_chunk(chunk_noise):
                latents_c = chunk_noise
                denoised_c, timestep_c = None, None
                for index, current_timestep in enumerate(self.denoising_step_list):
                    temp_ts = ((first_frame_mask_block[0, :, 0, ::2, ::2]) * current_timestep).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(self.frame_seq_length * current_num_frames - temp_ts.size(0)) * current_timestep
                    ])
                    timestep_c = temp_ts.unsqueeze(0).expand(batch_size, temp_ts.size(0))

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_c = self.generator(
                            noisy_image_or_video=latents_c,
                            conditional_dict=conditional_dict,
                            y=y_latents, y_camera=y_camera_latents,
                            timestep=timestep_c,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length)
                        next_timestep = self.denoising_step_list[index + 1]
                        next_timestep = next_timestep * torch.ones(
                            [batch_size, current_num_frames], device=noise.device, dtype=torch.long)
                        if i == 0:
                            next_timestep[:, 0] = 0
                        latents_c = self.scheduler.add_noise(
                            denoised_c.flatten(0, 1),
                            torch.randn_like(denoised_c.flatten(0, 1)),
                            next_timestep.flatten()
                        ).unflatten(0, denoised_c.shape[:2])
                        latents_c = latents_c * first_frame_mask_block + chunk_noise * (1 - first_frame_mask_block)
                    else:
                        _, denoised_c = self.generator(
                            noisy_image_or_video=latents_c,
                            conditional_dict=conditional_dict,
                            y=y_latents, y_camera=y_camera_latents,
                            timestep=timestep_c,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length)
                        denoised_c = denoised_c * first_frame_mask_block + chunk_noise * (1 - first_frame_mask_block)
                return denoised_c, timestep_c

            denoised_pred, timestep = _denoise_chunk(noisy_input)

            # Best-of-N re-roll (on-manifold closed loop): the scorer may regenerate this chunk
            # with fresh noise and return the best-scoring GENUINE model sample. Nothing
            # externally modified is ever committed. Skipped for chunk 0 (contains frame 0).
            if chunk_scorer is not None and i > 0:
                denoised_pred = chunk_scorer(
                    denoised_pred,
                    lambda: _denoise_chunk(torch.randn_like(noisy_input))[0],
                    chunk_index=i)

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Closed-loop hook: correct the chunk latent BEFORE it is committed to the
            # KV cache, so the correction propagates to all future chunks. Byte-identical
            # to the open-loop path when chunk_callback is None.
            if chunk_callback is not None:
                denoised_pred = chunk_callback(
                    denoised_pred, chunk_index=i, current_start_frame=current_start_frame)
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Rerun with context noise to update KV cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                y=y_latents, y_camera=y_camera_latents,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_times.append(block_start.elapsed_time(block_end))

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time
            print(f"Profiling: init={init_time:.0f}ms, diffusion={diffusion_time:.0f}ms, vae={vae_time:.0f}ms, total={total_time:.0f}ms")
            for i, bt in enumerate(block_times):
                print(f"  Block {i}: {bt:.0f}ms")

        return (video, output) if return_latents else video

    def _initialize_kv_cache(self, batch_size, dtype, device, num_frames=21):
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 18480

        num_head = 24
        self.kv_cache1 = [
            {
                "k": torch.zeros([batch_size, kv_cache_size, num_head, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_head, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(self.num_transformer_blocks)
        ]

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        num_head = 24
        self.crossattn_cache = [
            {
                "k": torch.zeros([batch_size, 512, num_head, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, num_head, 128], dtype=dtype, device=device),
                "is_init": False,
            }
            for _ in range(self.num_transformer_blocks)
        ]
