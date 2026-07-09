"""
validate_sensors.py — decide whether a cheap in-loop reflection signal is trustworthy
BEFORE building the closed loop. Runs entirely on OPEN-LOOP rollouts.

It answers, quantitatively, the question "will closed-loop beat open-loop?" via two
studies, each keyed to a criterion from the plan:

  STUDY 1 — CORRELATION (criteria 1,4)
    Over a horizon of open-loop chunks, measure cheap-sensor drift and gold-sensor
    drift (both = error vs the frame-0 anchor, per chunk). Report Spearman rank
    correlation. A cheap sensor is usable in-loop only if it rank-tracks a gold
    sensor (target Spearman >= 0.8). Also reports the per-chunk NOISE FLOOR (rerun
    same chunk under different seeds) so you can see SNR = drift / noise.

  STUDY 2 — COUNTERFACTUAL INJECTION (the decisive one; directionality criterion 2)
    Take drifted chunks. Apply the correction the controller WOULD apply (actuator A:
    moment-matching toward the frame-0 setpoint), decode, and measure the GOLD metric
    before vs after. If gold improves, the sensor->actuator->truth chain is sound and
    closed-loop WILL help — verified without ever closing the loop. If gold does NOT
    improve (or the cheap error drops while gold rises -> Goodhart/collocation), the
    signal is not good enough and no PID tuning will fix it.

Design notes:
  * We do NOT depend on the closed-loop pipeline patch. We drive the STOCK pipeline
    once per item to get open-loop latents (return_latents=True), then analyze offline.
  * The actuator here is a standalone copy of actuator (A) so the study is self-contained.
  * Everything that needs a face is guarded (ArcFace may return None).

Usage:
    CUDA_VISIBLE_DEVICES=0 python validate_sensors.py \
        --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
        --base_checkpoint_path /path/to/baseline.pt \
        --model_name /path/to/wan_models_root \
        --transformer_path /path/to/transformer_dir \
        --data_path configs/dreamx/eval.json \
        --output_folder sensor_validation/ \
        --num_output_frames 61 \
        --gold dino arcface \
        --cheap latent_moments latent_pooled4 \
        --noise_seeds 3
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

# reuse the stock inference machinery
from pipeline.pipeline_causal_camera import CausalCameraInferencePipeline
from utils.misc import set_seed
from utils.trajectory_processor import generate_trajectory_from_json, Camera
from utils.memory import gpu, get_cuda_free_memory_gb
from inference_ar_forcing import cam_params_to_prope_dict, load_pipeline

from sensor_zoo import CHEAP_SENSORS, load_gold_sensors, sensor_error


# ── frame indexing: map a latent-frame index to its position in the decoded video ──
# DreamX packs 1 + 4k pixel frames per latent frame chunking; we decode the whole
# clip once (cheap relative to generation) and index decoded frames by latent chunk.

def spearman(x, y):
    """Spearman rho without scipy: Pearson on ranks. Handles ties via argsort-of-argsort."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# ── actuator (A), standalone: moment-match a latent chunk toward a target moment vec ──

def apply_moment_correction(latent, target_moments, gain=1.0):
    """Push per-channel mean/std of `latent` toward `target_moments` (=[mean_c; std_c]).
    gain=1.0 == fully match the setpoint's moments (the strongest actuation)."""
    x = latent.float()
    C = x.shape[2]
    tgt_mean, tgt_std = target_moments[:C], target_moments[C:]
    cur_mean = x.mean(dim=(0, 1, 3, 4), keepdim=True)
    cur_std = x.std(dim=(0, 1, 3, 4), keepdim=True) + 1e-6
    tm = tgt_mean.view(1, 1, C, 1, 1)
    ts = tgt_std.view(1, 1, C, 1, 1)
    scale = 1.0 + gain * (ts / cur_std - 1.0)
    shifted = (x - cur_mean) * scale + cur_mean + gain * (tm - cur_mean)
    return shifted.to(latent.dtype)


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
    p.add_argument("--num_output_frames", type=int, default=61)   # longer -> more drift
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk_relative", action="store_true")
    p.add_argument("--gold", nargs="+", default=["dino"])
    p.add_argument("--cheap", nargs="+", default=list(CHEAP_SENSORS.keys()))
    p.add_argument("--noise_seeds", type=int, default=3,
                   help="reruns of chunk 1 under different seeds -> sensor noise floor")
    p.add_argument("--inject_gain", type=float, default=1.0)
    p.add_argument("--max_items", type=int, default=None)
    return p.parse_args()


def build_camera(item, num_pixel_frames, device, chunk_relative):
    action_seq = [a.lower() for a in item["action_seq"]]
    spec = list(zip(action_seq, item["action_speed_list"]))
    _, cam_np, _ = generate_trajectory_from_json(spec, num_frames=num_pixel_frames, return_cam_params=True)
    cams = [Camera(cam_np[i].tolist()) for i in range(cam_np.shape[0])]
    return cam_params_to_prope_dict(cams, device=device, chunk_relative=chunk_relative)


@torch.no_grad()
def rollout_latents(pipeline, item, args, device, transform, seed):
    """One open-loop rollout -> (video[B,T,H,W,C] in [0,1], latents[B,Tlat,48,44,80])."""
    set_seed(seed)
    pil = Image.open(item["image_path"]).convert("RGB")
    img_t = transform(pil).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
    initial_latent = pipeline.vae.encode_to_latent(img_t).to(device=device, dtype=torch.bfloat16)

    noise = torch.randn([1, args.num_output_frames, 48, 44, 80], device=device, dtype=torch.bfloat16)
    noise[:, 0] = initial_latent
    num_pixel_frames = (args.num_output_frames - 1) * 4 + 1
    y_camera = build_camera(item, num_pixel_frames, device, args.chunk_relative)

    video, latents = pipeline.inference(
        noise=noise, text_prompts=[item.get("caption", item.get("prompt", ""))],
        y=None, y_camera=y_camera, return_latents=True,
    )
    video = rearrange(video, "b t c h w -> b t h w c").cpu()   # [0,1]
    pipeline.vae.model.clear_cache()
    return video, latents


def decode_latent_frame(pipeline, latent_frame):
    """Decode a single latent frame [B,1,48,H,W] -> RGB [3,Hpix,Wpix] in [0,1]."""
    px = pipeline.vae.decode_to_pixel(latent_frame)            # [B,1,3,H,W] in [-1,1]
    pipeline.vae.model.clear_cache()
    px = (px * 0.5 + 0.5).clamp(0, 1)
    return px[0, :, 0] if px.shape[1] == 1 else px[0, :, -1]   # [3,H,W]


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

    gold = load_gold_sensors(args.gold, device)
    cheap = {k: CHEAP_SENSORS[k] for k in args.cheap}
    print(f"cheap: {list(cheap)} | gold: {list(gold)} | free VRAM {get_cuda_free_memory_gb(gpu):.0f}GB")

    transform = transforms.Compose([
        transforms.Resize((704, 1280)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    with open(args.data_path) as f:
        items = json.load(f)
    if args.max_items:
        items = items[: args.max_items]

    report = {"items": [], "config": {"num_output_frames": args.num_output_frames,
                                       "inject_gain": args.inject_gain}}

    for idx, item in enumerate(items):
        print(f"\n[{idx}] {item['image_path']}")
        video, latents = rollout_latents(pipeline, item, args, device, transform, args.seed)
        B, Tlat = latents.shape[0], latents.shape[1]

        # anchor = frame 0
        anchor_frame = decode_latent_frame(pipeline, latents[:, 0:1])
        anchor_moments = CHEAP_SENSORS["latent_moments"](latents[:, 0:1])
        setpoint_moments = {k: cheap[k](latents[:, 0:1]) for k in cheap}
        gold_ref = {}
        for g, s in gold.items():
            if hasattr(s, "set_reference"):
                s.set_reference(anchor_frame)
                gold_ref[g] = None                       # scalar-distance style
            else:
                r = s(anchor_frame)
                gold_ref[g] = r

        # ── STUDY 1: per-chunk drift, cheap vs gold ──
        cheap_drift = {k: [] for k in cheap}
        gold_drift = {g: [] for g in gold}
        for t in range(1, Tlat):
            lf = latents[:, t:t + 1]
            for k in cheap:
                cheap_drift[k].append(sensor_error(cheap[k](lf), setpoint_moments[k]))
            frame = decode_latent_frame(pipeline, lf)
            for g, s in gold.items():
                r = s(frame)
                if r is None:                            # face dropout
                    gold_drift[g].append(float("nan")); continue
                if gold_ref[g] is None:                  # scalar-distance sensor
                    gold_drift[g].append(float(r.item()))
                else:
                    gold_drift[g].append(sensor_error(r, gold_ref[g]))

        corr = {f"{k}~{g}": spearman(cheap_drift[k], gold_drift[g])
                for k in cheap for g in gold}

        # ── noise floor: rerun chunk 1 under different seeds ──
        noise_floor = {k: [] for k in cheap}
        base_lf1 = latents[:, 1:2]
        for s_i in range(1, args.noise_seeds):
            v2, l2 = rollout_latents(pipeline, item, args, device, transform, args.seed + 1000 * s_i)
            for k in cheap:
                noise_floor[k].append(sensor_error(cheap[k](l2[:, 1:2]), cheap[k](base_lf1)))

        # ── STUDY 2: counterfactual injection on the LAST (most drifted) chunk ──
        inj = {}
        drift_lf = latents[:, Tlat - 1:Tlat]
        for g, s in gold.items():
            # gold error of the raw drifted chunk
            f_before = decode_latent_frame(pipeline, drift_lf)
            r_b = s(f_before)
            # apply actuator (A) toward the moment setpoint, decode, re-measure gold
            corrected = apply_moment_correction(drift_lf, anchor_moments, gain=args.inject_gain)
            f_after = decode_latent_frame(pipeline, corrected)
            r_a = s(f_after)
            if r_b is None or r_a is None:
                inj[g] = None; continue
            if gold_ref[g] is None:
                e_before, e_after = float(r_b.item()), float(r_a.item())
            else:
                e_before = sensor_error(r_b, gold_ref[g])
                e_after = sensor_error(r_a, gold_ref[g])
            inj[g] = {"gold_before": e_before, "gold_after": e_after,
                      "improved": e_after < e_before, "delta": e_before - e_after}

        # cheap error before/after injection (should always drop -> if it drops but
        # gold does NOT, that's the Goodhart signature)
        cheap_inj = {k: {
            "before": sensor_error(cheap[k](drift_lf), setpoint_moments[k]),
            "after": sensor_error(cheap[k](apply_moment_correction(drift_lf, anchor_moments,
                                                                   gain=args.inject_gain)),
                                  setpoint_moments[k]),
        } for k in cheap}

        rec = {
            "image_path": item["image_path"],
            "spearman": corr,
            "cheap_drift_last": {k: cheap_drift[k][-1] for k in cheap},
            "gold_drift_last": {g: gold_drift[g][-1] for g in gold},
            "noise_floor_mean": {k: float(np.nanmean(noise_floor[k])) if noise_floor[k] else None
                                 for k in cheap},
            "snr_last": {k: (cheap_drift[k][-1] / np.nanmean(noise_floor[k]))
                         if noise_floor[k] and np.nanmean(noise_floor[k]) > 0 else None
                         for k in cheap},
            "injection_gold": inj,
            "injection_cheap": cheap_inj,
        }
        report["items"].append(rec)
        print(f"    spearman: { {k: round(v,2) for k,v in corr.items()} }")
        print(f"    injection gold improved: { {g: (inj[g]['improved'] if inj[g] else None) for g in gold} }")

    # ── aggregate verdict ──
    def agg(path_fn):
        vals = [path_fn(r) for r in report["items"]]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    verdict = {}
    for k in cheap:
        for g in gold:
            key = f"{k}~{g}"
            verdict[f"spearman_{key}"] = agg(lambda r: r["spearman"].get(key))
    for g in gold:
        imps = [r["injection_gold"][g]["improved"] for r in report["items"]
                if r["injection_gold"].get(g)]
        verdict[f"inject_gold_improve_rate_{g}"] = float(np.mean(imps)) if imps else None
    for k in cheap:
        verdict[f"snr_{k}"] = agg(lambda r: r["snr_last"].get(k))
    report["verdict"] = verdict

    out = os.path.join(args.output_folder, "sensor_validation_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60 + "\nVERDICT\n" + "=" * 60)
    for k, v in verdict.items():
        print(f"  {k:42s}: {round(v,3) if v is not None else 'n/a'}")
    print("\nDecision rules:")
    print("  * cheap sensor USABLE in-loop iff spearman ~ gold >= 0.80 AND snr >= ~3")
    print("  * closed-loop WILL beat open-loop iff inject_gold_improve_rate >= ~0.8")
    print("  * if cheap error drops on injection but gold does NOT improve -> Goodhart,")
    print("    promote the gold sensor to the in-loop sensor (eat the decode cost).")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
