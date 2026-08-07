import os
import sys
import argparse
import json
import math
import numpy as np
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from transformers import AutoTokenizer

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from dist import set_multi_gpus_devices, shard_model
from models import (AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer,
                    WanT5EncoderModel)
from models import Wan2_2Transformer3DModel


from pipeline.pipeline_dreamxworld import Wan2_2_CameraPipeline
from utils.fp8_optimization import (convert_model_weight_to_float8,
                                    convert_weight_dtype_wrapper,
                                    replace_parameters_by_name)
from utils.lora_utils import merge_lora, unmerge_lora
from utils.utils import (filter_kwargs, save_videos_grid)
from utils.fm_solvers import FlowDPMSolverMultistepScheduler
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.inference_utils import ActionToPoseFromID, GetPoseEmbedsFromPosesPrope

import torch.nn.functional as F
import torchvision.transforms as transforms


def print_info(*args, **kwargs):
    """Print information, handling distributed training."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Video generation with camera control")
    
    # Model and config paths
    parser.add_argument("--config_path", type=str, default="configs/wan2.2/wan_ti2v_5b.yaml",
                        help="Path to model config file")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Path to model directory")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to pretrained transformer checkpoint (low noise model)")
    parser.add_argument("--transformer_high_path", type=str, default=None,
                        help="Path to pretrained transformer checkpoint (high noise model)")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="Path to pretrained VAE checkpoint")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA checkpoint (low noise model)")
    parser.add_argument("--lora_high_path", type=str, default=None,
                        help="Path to LoRA checkpoint (high noise model)")
    
    # Input/Output parameters
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing images, or path to a single image file, or path to a JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for generated videos")
    
    
    # Generation parameters
    parser.add_argument("--sample_size", type=int, nargs=2, default=[480, 832],
                        help="Video sample size [height, width]")
    parser.add_argument("--video_length", type=int, default=161,
                        help="Number of frames to generate")
    parser.add_argument("--fps", type=int, default=16,
                        help="Frames per second")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=43,
                        help="Random seed")
    parser.add_argument("--negative_prompt", type=str, 
                        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        help="Negative prompt")
    
    # Camera control parameters
    parser.add_argument("--cam_method", type=str, default="prope",
                        choices=["prope", "plucker"],
                        help="Camera action type")
    parser.add_argument("--add_control_adapter", action="store_true", default=True,
                        help="Use control adapter")
 
    
    # Sampler parameters
    parser.add_argument("--sampler_name", type=str, default="Flow",
                        choices=["Flow", "Flow_Unipc", "Flow_DPM++"],
                        help="Sampler type")
    parser.add_argument("--shift", type=float, default=3.0,
                        help="Noise schedule shift parameter")
    
    # Memory and optimization
    parser.add_argument("--GPU_memory_mode", type=str, default=None,
                        choices=["model_full_load", "model_full_load_and_qfloat8", 
                                "model_cpu_offload", "model_cpu_offload_and_qfloat8", 
                                "sequential_cpu_offload"],
                        help="GPU memory management mode")
    parser.add_argument("--ulysses_degree", type=int, default=1,
                        help="Ulysses degree for multi-GPU")
    parser.add_argument("--ring_degree", type=int, default=1,
                        help="Ring degree for multi-GPU")
    parser.add_argument("--fsdp_dit", action="store_true",
                        help="Use FSDP for DiT")
    parser.add_argument("--fsdp_text_encoder", action="store_true", default=False,
                        help="Use FSDP for text encoder")
    parser.add_argument("--compile_dit", action="store_true",
                        help="Compile DiT for speedup")
    
    
    # LoRA parameters
    parser.add_argument("--lora_weight", type=float, default=0.55,
                        help="LoRA weight (low noise model)")
    parser.add_argument("--lora_high_weight", type=float, default=0.55,
                        help="LoRA weight (high noise model)")
    
    # Data type
    parser.add_argument("--weight_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Weight data type")

    args = parser.parse_args()
    
    return args

def setup_models(args):
    """Setup and load all models."""
    device = set_multi_gpus_devices(args.ulysses_degree, args.ring_degree)
    config = OmegaConf.load(args.config_path)
    
    # Set weight dtype
    if args.weight_dtype == "float16":
        weight_dtype = torch.float16
    elif args.weight_dtype == "bfloat16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Get boundary from config
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

    # Load transformer (low noise model)
    config['transformer_additional_kwargs']['cam_method'] = args.cam_method
    config['transformer_additional_kwargs']['add_control_adapter'] = args.add_control_adapter

    if args.transformer_path is not None:
        print_info(f"Loading transformer from checkpoint: {args.transformer_path}")
        transformer = Wan2_2Transformer3DModel.from_pretrained(
            args.transformer_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            # low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        
    
    if args.transformer_high_path is not None:
        print_info(f"Loading transformer from checkpoint: {args.transformer_high_path}")
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            args.transformer_high_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            # low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None
        

    # Load VAE with type selection
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if args.vae_path is not None:
        print_info(f"Loading VAE from checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        print_info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Load text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Setup scheduler
    scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }
    Chosen_Scheduler = scheduler_dict[args.sampler_name]
    
    if args.sampler_name == "Flow_Unipc" or args.sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Create pipeline
    pipeline = Wan2_2_CameraPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    # Multi-GPU setup
    if args.ulysses_degree > 1 or args.ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if transformer_2 is not None:
            transformer_2.enable_multi_gpus_inference()
        if args.fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print_info("Added FSDP DIT")
        if args.fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print_info("Added FSDP TEXT ENCODER")

    # Compile if requested
    if args.compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        if transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
        print_info("Added Compile")

    # Memory management
    if args.GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        if transformer_2 is not None:
            replace_parameters_by_name(transformer_2, ["modulation",], device=device)
            transformer_2.freqs = transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif args.GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        if transformer_2 is not None:
            convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
            convert_weight_dtype_wrapper(transformer_2, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)


    return pipeline, device, weight_dtype, config, vae, boundary

def get_camera_sequence(action_ids, action_speed_list, args):
    """Generate camera control sequence from action IDs."""
   
    duration = math.ceil(args.video_length / len(action_ids))
    total_pose = ActionToPoseFromID(action_ids, action_speed_list, duration=duration)
    total_pose = total_pose[:args.video_length]
    
    control_camera_video, _ = GetPoseEmbedsFromPosesPrope(
        total_pose, args.sample_size[0], args.sample_size[1], len(total_pose), False, 0
    )
    
    return control_camera_video, len(total_pose)


def process_inference_from_json(args, pipeline, device, vae, boundary):
    """Process inference for all images in input directory."""

    with open(args.input_dir, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    
    print_info(f"🎯 Found {len(items)} images to process")
    print_info(f"📂 Input directory: {args.input_dir}")
    print_info(f"📂 Output directory: {args.output_dir}")
    print_info("=" * 80)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    

    # Process each image
    for idx, item in enumerate(items):

        start_image_path = item['image_path']
        prompt = item['caption'].replace('\n', ' ').replace('\r', ' ').strip()

        file_name = os.path.basename(start_image_path)
        name, _ = os.path.splitext(file_name)
        
        # Prepare Camera
        action_seq = item.get('action_seq')
        action_speed_list = item.get('action_speed_list')
        control_camera_video, _ = get_camera_sequence(action_seq, action_speed_list, args)

        action_name = '_'.join(action_seq) if action_seq else 'default'
        video_name = name + '_' + action_name + '.mp4'
        video_path = os.path.join(args.output_dir, video_name)

   
        print_info(f"\n🎬 Processing image [{idx+1}/{len(items)}]: {file_name}")
        print_info(f"📝 Prompt: {prompt}")
        
        # Calculate video length based on VAE temporal compression
        video_length = args.video_length
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
        
        
        print_info(f"⚙️  Generation parameters:")
        print_info(f"   - Steps: {args.num_inference_steps}")
        print_info(f"   - Guidance scale: {args.guidance_scale}")
        print_info(f"   - Resolution: {args.sample_size[0]}x{args.sample_size[1]}")
        print_info(f"   - Video length: {video_length} frames")
        print_info(f"   - FPS: {args.fps}")
        print_info(f"   - Boundary: {boundary}")
        print_info("🚀 Starting inference...")
        
        # Generate video
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
    
        with torch.no_grad():
            sample = pipeline(
                prompt,
                num_frames=video_length,
                negative_prompt=args.negative_prompt,
                height=args.sample_size[0],
                width=args.sample_size[1],
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                start_image=start_image_path,
                control_camera_video=control_camera_video,
                boundary=boundary,
                shift=args.shift
            ).videos
        
        # Save video
        if args.ulysses_degree * args.ring_degree > 1:
            if dist.get_rank() == 0:
                save_videos_grid(sample, video_path, fps=args.fps)
        else:
            save_videos_grid(sample, video_path, fps=args.fps)
        
        print_info(f"🎉 Video saved to: {video_path}")
    
    print_info("\n" + "=" * 80)
    print_info("✅ All videos generated successfully!")


def main():
    """Main function."""
    args = parse_args()
    
    print_info("=" * 80)
    print_info("🚀 Wan2.2 Video Generation with Camera Control")
    print_info("=" * 80)
    print_info("\n📋 Configuration:")
    print_info(f"   - Model: {args.model_name}")
    print_info(f"   - Config: {args.config_path}")
    print_info(f"   - Input: {args.input_dir}")
    print_info(f"   - Output: {args.output_dir}")
    print_info(f"   - GPU Memory Mode: {args.GPU_memory_mode}")
    print_info(f"   - Weight dtype: {args.weight_dtype}")
    print_info("")
    
    print_info("🔧 Setting up models...")
    pipeline, device, weight_dtype, config, vae, boundary = setup_models(args)
    
    # Load LoRA if specified
    if args.lora_path is not None:
        print_info(f"📦 Loading LoRA from: {args.lora_path}")
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)
        if pipeline.transformer_2 is not None and args.lora_high_path is not None:
            pipeline = merge_lora(pipeline, args.lora_high_path, args.lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
    
    print_info("✅ Models loaded successfully!\n")
    

    # Process inference
    process_inference_from_json(args, pipeline, device, vae, boundary)


    # Unmerge LoRA after inference
    if args.lora_path is not None:
        pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)
        if pipeline.transformer_2 is not None and args.lora_high_path is not None:
            pipeline = unmerge_lora(pipeline, args.lora_high_path, args.lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")
   

if __name__ == "__main__":
    main()
