"""
demo_closed_loop.py — head-to-head open-loop vs closed-loop DreamX-World generation.

For each item, generates the SAME rollout (identical seed/noise/camera/prompt) once per
mode:
  * open : chunk_callback = None                       (stock model)
  * pid  : leaky-PID moment control (v1, no deadband)
  * pid2 : leaky-PID + deadband, I-dominant (v2 — does not fight reasonable drift/motion)
  * i / p: integral-only / proportional-only ablations
Postprocess color correction is OFF for both, so we measure the model's true drift.

Evaluation (gold, controller never sees it): DINOv3 drift per latent frame =
1 - cos(dino(frame_t), dino(frame_0)). NOTE: distance-from-frame-0 conflates unwanted drift
with legitimate scene change; use fair_eval.py (motion energy + tOF) alongside this.

Outputs per item under --output_folder:
  <name>_<mode>.mp4, <name>_sidebyside.mp4 (all modes), <name>_drift.json, <name>_drift.png
plus summary_shard<id>.json.
"""

import argparse
import json
import os

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.io import write_video

from utils.misc import set_seed
from utils.trajectory_processor import generate_trajectory_from_json, Camera
from utils.memory import gpu, get_cuda_free_memory_gb
from inference_ar_forcing import cam_params_to_prope_dict, load_pipeline

from sensor_zoo import load_gold_sensors, sensor_error
from dreamx_closed_loop import DreamXClosedLoop, GoldRerollScorer, PIDRerollScorer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_path", required=True)
    p.add_argument("--model_name", default=None)
    p.add_argument("--transformer_path", default=None)
    p.add_argument("--vae_path", default=None)
    p.add_argument("--base_checkpoint_path", default=None)
    p.add_argument("--checkpoint_path", default=None)
    p.add_argument("--lora_ckpt", default=None)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_folder", required=True)
    p.add_argument("--num_output_frames", type=int, default=63)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--chunk_relative", action="store_true")
    p.add_argument("--modes", nargs="+", default=["open", "pid", "pid2"],
                   help="subset of: open p i pid pid2 ctxnoise both")
    p.add_argument("--kp", type=float, default=0.2)
    p.add_argument("--ki", type=float, default=0.1)
    p.add_argument("--leak", type=float, default=0.85)
    p.add_argument("--gain_max", type=float, default=0.3)
    # v2 (deadband + I-dominant): does not fight reasonable drift/motion
    p.add_argument("--kp2", type=float, default=0.05)
    p.add_argument("--ki2", type=float, default=0.12)
    p.add_argument("--leak2", type=float, default=0.9)
    p.add_argument("--deadband2", type=float, default=0.10)
    # actuator B (adaptive context_noise): ctx = clamp(base + k*max(0,drift-db), base, max)
    p.add_argument("--ctx_base", type=float, default=0.1)
    p.add_argument("--ctx_k", type=float, default=1.0)
    p.add_argument("--ctx_max", type=float, default=0.4)
    p.add_argument("--ctx_deadband", type=float, default=0.05)
    p.add_argument("--sink_size", type=int, default=None,
                   help="override attention-sink size at model load (ablation; None=default 3)")
    # actuator D (gold-gated shift)
    p.add_argument("--decode_every", type=int, default=3)
    p.add_argument("--gold_thresh", type=float, default=0.15)
    p.add_argument("--grad_step", type=float, default=0.05)
    p.add_argument("--ffwd_gain", type=float, default=0.5)
    p.add_argument("--smooth_beta", type=float, default=0.7)
    p.add_argument("--smooth_gain", type=float, default=0.3)
    # best-of-N re-roll (on-manifold selection)
    p.add_argument("--reroll_n", type=int, default=2, help="extra candidates per gated chunk")
    p.add_argument("--reroll_thresh", type=float, default=0.2, help="gold drift gate for re-roll")
    # PID-scheduled re-roll (PI law allocates candidate budget per chunk)
    p.add_argument("--pr_kp", type=float, default=4.0)
    p.add_argument("--pr_ki", type=float, default=2.0)
    p.add_argument("--pr_leak", type=float, default=0.9)
    p.add_argument("--pr_deadband", type=float, default=0.2)
    p.add_argument("--pr_nmax", type=int, default=7)
    p.add_argument("--drift_stride", type=int, default=1,
                   help="evaluate DINOv3 drift every k latent frames (cheaper for long videos)")
    p.add_argument("--max_items", type=int, default=None)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    return p.parse_args()


def build_camera(item, num_pixel_frames, device, chunk_relative):
    spec = list(zip([a.lower() for a in item["action_seq"]], item["action_speed_list"]))
    _, cam_np, _ = generate_trajectory_from_json(spec, num_frames=num_pixel_frames, return_cam_params=True)
    cams = [Camera(cam_np[i].tolist()) for i in range(cam_np.shape[0])]
    return cam_params_to_prope_dict(cams, device=device, chunk_relative=chunk_relative)


def make_controller(mode, initial_latent, args, pipeline, dino=None):
    if mode == "open":
        return None
    if mode == "pid2":
        return DreamXClosedLoop(initial_latent, mode="pid", kp=args.kp2, ki=args.ki2,
                                leak=args.leak2, gain_max=args.gain_max, deadband=args.deadband2)
    if mode == "ctxnoise":
        return DreamXClosedLoop(initial_latent, actuator="ctxnoise", args_ref=pipeline.args,
                                ctx_base=args.ctx_base, ctx_k=args.ctx_k, ctx_max=args.ctx_max,
                                ctx_deadband=args.ctx_deadband)
    if mode == "both":
        return DreamXClosedLoop(initial_latent, mode="pid", actuator="both", args_ref=pipeline.args,
                                kp=args.kp2, ki=args.ki2, leak=args.leak2, gain_max=args.gain_max,
                                deadband=args.deadband2, ctx_base=args.ctx_base, ctx_k=args.ctx_k,
                                ctx_max=args.ctx_max, ctx_deadband=args.ctx_deadband)
    if mode == "shift":
        return DreamXClosedLoop(initial_latent, mode="pid", actuator="shift",
                                kp=args.kp2, ki=args.ki2, leak=args.leak2,
                                gain_max=args.gain_max, deadband=args.deadband2)
    if mode == "smooth":
        return DreamXClosedLoop(initial_latent, actuator="smooth",
                                smooth_beta=args.smooth_beta, smooth_gain=args.smooth_gain)
    if mode == "gold_shift":
        return DreamXClosedLoop(initial_latent, mode="pid", actuator="gold_shift",
                                kp=args.kp2, ki=args.ki2, leak=args.leak2, gain_max=args.gain_max,
                                decode_fn=lambda lf: decode_latent_frame(pipeline, lf), dino=dino,
                                decode_every=args.decode_every, gold_thresh=args.gold_thresh)
    if mode == "gold_ctx":
        return DreamXClosedLoop(initial_latent, mode="pid", actuator="gold_ctx", args_ref=pipeline.args,
                                kp=args.kp2, ki=args.ki2, leak=args.leak2, gain_max=args.gain_max,
                                decode_fn=lambda lf: decode_latent_frame(pipeline, lf), dino=dino,
                                decode_every=args.decode_every, gold_thresh=args.gold_thresh,
                                ctx_base=args.ctx_base, ctx_k=args.ctx_k, ctx_max=args.ctx_max,
                                ctx_deadband=args.ctx_deadband)
    if mode == "gold_grad":
        def _decode_grad(lat):
            px = pipeline.vae.decode_to_pixel(lat)          # grad-enabled (caller uses enable_grad)
            px = (px * 0.5 + 0.5).clamp(0, 1)
            return px[0, :, 0] if px.shape[1] == 1 else px[0, :, -1]
        return DreamXClosedLoop(initial_latent, mode="pid", actuator="gold_grad", args_ref=pipeline.args,
                                gain_max=args.gain_max, dino=dino, decode_grad_fn=_decode_grad,
                                decode_every=args.decode_every, gold_thresh=args.gold_thresh,
                                grad_step=args.grad_step, ctx_base=args.ctx_base, ctx_k=args.ctx_k,
                                ctx_max=args.ctx_max, ctx_deadband=args.ctx_deadband)
    return DreamXClosedLoop(initial_latent, mode=mode, kp=args.kp, ki=args.ki,
                            leak=args.leak, gain_max=args.gain_max)


@torch.no_grad()
def rollout(pipeline, item, args, device, transform, mode, dino=None):
    """Deterministic rollout: same seed -> same noise/RNG stream for every mode."""
    set_seed(args.seed)
    pil = Image.open(item["image_path"]).convert("RGB")
    img_t = transform(pil).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
    initial_latent = pipeline.vae.encode_to_latent(img_t).to(device=device, dtype=torch.bfloat16)

    noise = torch.randn([1, args.num_output_frames, 48, 44, 80], device=device, dtype=torch.bfloat16)
    noise[:, 0] = initial_latent
    num_pixel_frames = (args.num_output_frames - 1) * 4 + 1
    y_camera = build_camera(item, num_pixel_frames, device, args.chunk_relative)

    # reset context_noise to baseline each rollout so actuator-B state doesn't leak across modes/items
    pipeline.args.context_noise = args.ctx_base
    # feedforward mode: run OPEN, then post-correct the output latents toward frame-0 appearance
    # WITHOUT feeding anything back to the KV cache (pure feedforward, not closed-loop). Isolates
    # whether appearance correction OUTSIDE the feedback loop preserves motion.
    if mode == "ffwd":
        video, latents = pipeline.inference(
            noise=noise, text_prompts=[item.get("caption", item.get("prompt", ""))],
            y=None, y_camera=y_camera, return_latents=True, chunk_callback=None)
        pipeline.vae.model.clear_cache()
        latc = ffwd_correct(latents, initial_latent, gain=args.ffwd_gain, deadband=args.deadband2)
        vid = pipeline.vae.decode_to_pixel(latc)
        vid = (vid * 0.5 + 0.5).clamp(0, 1)
        pipeline.vae.model.clear_cache()
        return vid.float().cpu(), latc, None
    # best-of-N re-roll mode: on-manifold closed loop via SELECTION (chunk_scorer), not actuation
    if mode in ("reroll", "pid_reroll"):
        if mode == "reroll":
            scorer = GoldRerollScorer(initial_latent,
                                      lambda lf: decode_latent_frame(pipeline, lf), dino,
                                      gold_thresh=args.reroll_thresh, n_extra=args.reroll_n)
        else:
            scorer = PIDRerollScorer(initial_latent,
                                     lambda lf: decode_latent_frame(pipeline, lf), dino,
                                     kp=args.pr_kp, ki=args.pr_ki, leak=args.pr_leak,
                                     deadband=args.pr_deadband, n_max=args.pr_nmax)
        video, latents = pipeline.inference(
            noise=noise, text_prompts=[item.get("caption", item.get("prompt", ""))],
            y=None, y_camera=y_camera, return_latents=True, chunk_scorer=scorer)
        pipeline.vae.model.clear_cache()
        print(f"      {mode} stats: {scorer.stats()}")
        return video.float().cpu(), latents, scorer
    ctrl = make_controller(mode, initial_latent, args, pipeline, dino=dino)
    video, latents = pipeline.inference(
        noise=noise, text_prompts=[item.get("caption", item.get("prompt", ""))],
        y=None, y_camera=y_camera, return_latents=True, chunk_callback=ctrl)
    pipeline.vae.model.clear_cache()
    return video.float().cpu(), latents, ctrl


def ffwd_correct(latents, initial_latent, gain=0.5, deadband=0.10):
    """Feedforward: shift each output frame's per-channel mean toward frame-0's per-channel mean,
    proportional to that frame's drift beyond a deadband. Never touches the KV cache. Preserves
    per-frame/per-pixel structure (additive DC shift only)."""
    x = latents.float()                                     # [B,T,C,H,W]
    r_mean = initial_latent.float().mean(dim=(0, 1, 3, 4))  # [C] frame-0 per-channel mean
    r_std = initial_latent.float().std(dim=(0, 1, 3, 4))
    C = x.shape[2]
    out = x.clone()
    rv = torch.cat([r_mean, r_std])
    for t in range(1, x.shape[1]):                          # leave frame 0 untouched
        ft = x[:, t:t + 1]
        m = ft.mean(dim=(0, 1, 3, 4)); s = ft.std(dim=(0, 1, 3, 4))
        drift = float(1.0 - torch.nn.functional.cosine_similarity(torch.cat([m, s]), rv, dim=0, eps=1e-8))
        if drift < deadband:
            continue
        out[:, t] = ft[:, 0] + (gain * (r_mean - m)).view(1, C, 1, 1)
    return out.to(latents.dtype)


def decode_latent_frame(pipeline, latent_frame):
    px = pipeline.vae.decode_to_pixel(latent_frame)
    pipeline.vae.model.clear_cache()
    px = (px * 0.5 + 0.5).clamp(0, 1)
    return px[0, :, 0] if px.shape[1] == 1 else px[0, :, -1]


@torch.no_grad()
def dino_drift(pipeline, latents, dino, stride=1):
    """1 - cos(dino(frame_t), dino(frame_0)); evaluate every `stride` latent frames."""
    anchor = dino(decode_latent_frame(pipeline, latents[:, 0:1]))
    idxs = list(range(0, latents.shape[1], stride))
    if idxs[-1] != latents.shape[1] - 1:
        idxs.append(latents.shape[1] - 1)              # always include the last (most-drifted) frame
    out = []
    for t in idxs:
        emb = dino(decode_latent_frame(pipeline, latents[:, t:t + 1]))
        out.append(sensor_error(emb, anchor))
    return out


def save_video(path, video_bchw, fps):
    v = rearrange(video_bchw, "b t c h w -> b t h w c")[0]
    write_video(path, (v * 255).clamp(0, 255).to(torch.uint8), fps=fps)


def slope(y):
    y = np.asarray(y, float)
    if len(y) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])


def main():
    args = parse_args()
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    os.makedirs(args.output_folder, exist_ok=True)

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(os.path.join(os.path.dirname(args.config_path), "default_config.yaml"))
    config = OmegaConf.merge(default_config, config)

    pipeline = load_pipeline(args, config, device).to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device); pipeline.generator.to(device); pipeline.vae.to(device)
    dino = load_gold_sensors(["dino3"], device)["dino3"]
    print(f"modes={args.modes} frames={args.num_output_frames} "
          f"v1(kp={args.kp},ki={args.ki}) v2(kp={args.kp2},ki={args.ki2},deadband={args.deadband2}) "
          f"free VRAM {get_cuda_free_memory_gb(gpu):.0f}GB")

    transform = transforms.Compose([
        transforms.Resize((704, 1280)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    with open(args.data_path) as f:
        items = json.load(f)
    if args.max_items:
        items = items[: args.max_items]
    items = items[args.shard_id::args.num_shards]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(items)} items")

    summary = {"config": vars(args), "items": []}
    for idx, item in enumerate(items):
        name = os.path.splitext(os.path.basename(item["image_path"]))[0]
        print(f"\n[{idx}] {name}")
        videos, drifts, gates = {}, {}, {}
        for mode in args.modes:
            video, latents, ctrl = rollout(pipeline, item, args, device, transform, mode, dino=dino)
            videos[mode] = video
            drifts[mode] = dino_drift(pipeline, latents, dino, stride=args.drift_stride)
            gates[mode] = (ctrl.gate_log if ctrl else [])
            save_video(os.path.join(args.output_folder, f"{name}_{mode}.mp4"), video, args.fps)
            g = f" gated={int(sum(gates[mode]))}/{len(gates[mode])}" if gates[mode] else ""
            print(f"    {mode:5s}: dino drift mean={np.mean(drifts[mode]):.4f} "
                  f"final={drifts[mode][-1]:.4f} slope={slope(drifts[mode]):.5f}{g}")

        if len(args.modes) >= 2:
            t = min(v.shape[1] for v in videos.values())
            sbs = torch.cat([videos[m][:, :t] for m in args.modes], dim=-1)   # concat along width
            save_video(os.path.join(args.output_folder, f"{name}_sidebyside.mp4"), sbs, args.fps)

        with open(os.path.join(args.output_folder, f"{name}_drift.json"), "w") as f:
            json.dump(drifts, f, indent=2)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4.5))
            for mode in args.modes:
                plt.plot(drifts[mode], label=mode, marker=".")
            plt.xlabel(f"eval frame (stride {args.drift_stride})")
            plt.ylabel("DINOv3 drift  (1 - cos vs frame 0)")
            plt.title(f"{name}  —  appearance drift vs time"); plt.legend(); plt.grid(alpha=.3)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_folder, f"{name}_drift.png"), dpi=110)
            plt.close()
        except Exception as e:
            print(f"    (plot skipped: {e})")

        rec = {"name": name}
        for mode in args.modes:
            rec[mode] = {"mean": float(np.mean(drifts[mode])), "final": float(drifts[mode][-1]),
                         "slope": slope(drifts[mode])}
        summary["items"].append(rec)

    with open(os.path.join(args.output_folder, f"summary_shard{args.shard_id}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.output_folder}/summary_shard{args.shard_id}.json")


if __name__ == "__main__":
    main()
