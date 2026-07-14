"""
window_eval.py — windowed / local DINOv3 drift on already-generated videos.

Fixed frame-0 drift conflates unwanted appearance drift with cumulative legitimate scene
change. This recomputes DINOv3 appearance change against a SHIFTING reference:

  drift0[t]      = 1 - cos(e_t, e_0)                 (fixed frame-0, for reference)
  driftLag_L[t]  = 1 - cos(e_t, e_{t-L})             (change over the last L embed-steps)
  driftWmean_L[t]= 1 - cos(e_t, mean(e_{t-L..t-1}))  (change vs recent windowed mean)

Reads the mp4s written by demo_closed_loop.py and re-embeds frames with DINOv3 (no
regeneration). Shardable across GPUs. NOTE: windowed drift removes the cumulative-motion
confound but is STILL minimized by a frozen video, so read it with motion (fair_eval.py).
"""

import argparse, glob, json, os
import numpy as np
import torch
from torchvision.io import read_video

from sensor_zoo import load_gold_sensors, sensor_error


@torch.no_grad()
def embed_video(path, dino, stride=2):
    v, _, _ = read_video(path, pts_unit="sec", output_format="TCHW")   # uint8 [T,C,H,W]
    idx = list(range(0, v.shape[0], stride))
    embs = []
    for t in idx:
        frame = v[t].float() / 255.0
        embs.append(dino(frame))
    return torch.stack(embs)   # [N, D]


def drift_series(embs):
    e0 = embs[0]
    d0 = [float(1.0 - torch.nn.functional.cosine_similarity(embs[i], e0, dim=0, eps=1e-8))
          for i in range(embs.shape[0])]
    return d0


def lag_series(embs, L):
    out = []
    for i in range(L, embs.shape[0]):
        out.append(float(1.0 - torch.nn.functional.cosine_similarity(embs[i], embs[i - L], dim=0, eps=1e-8)))
    return out


def wmean_series(embs, L):
    out = []
    for i in range(L, embs.shape[0]):
        ref = embs[i - L:i].mean(0)
        out.append(float(1.0 - torch.nn.functional.cosine_similarity(embs[i], ref, dim=0, eps=1e-8)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--modes", nargs="+", default=["open", "pid", "pid2"])
    ap.add_argument("--stride", type=int, default=2, help="embed every k-th pixel frame")
    ap.add_argument("--lags", nargs="+", type=int, default=[4, 8], help="window sizes in embed-steps")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    dino = load_gold_sensors(["dino3"], device)["dino3"]

    names = sorted(os.path.basename(f).replace("_drift.json", "")
                   for f in glob.glob(os.path.join(args.folder, "*_drift.json")))
    names = names[args.shard_id::args.num_shards]

    rows = []
    for name in names:
        row = {"name": name}
        for m in args.modes:
            mp4 = os.path.join(args.folder, f"{name}_{m}.mp4")
            if not os.path.exists(mp4):
                continue
            embs = embed_video(mp4, dino, stride=args.stride)
            d0 = drift_series(embs)
            rec = {"drift0_final": d0[-1], "drift0_mean": float(np.mean(d0))}
            for L in args.lags:
                rec[f"lag{L}_mean"] = float(np.mean(lag_series(embs, L)))
                rec[f"wmean{L}_mean"] = float(np.mean(wmean_series(embs, L)))
            row[m] = rec
        rows.append(row)
        print(f"[{name[:26]:26}] " + "  ".join(
            f"{m}: d0={row[m]['drift0_final']:.3f} lag{args.lags[0]}={row[m][f'lag{args.lags[0]}_mean']:.3f}"
            for m in args.modes if m in row), flush=True)

    with open(os.path.join(args.folder, f"window_eval_shard{args.shard_id}.json"), "w") as f:
        json.dump({"per_item": rows, "lags": args.lags}, f, indent=2)
    print(f"wrote window_eval_shard{args.shard_id}.json")


if __name__ == "__main__":
    main()
