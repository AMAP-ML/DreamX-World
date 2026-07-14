"""
fair_eval.py — motion-fair evaluation of open vs closed-loop videos.

Frame-0 DINOv3 drift conflates unwanted appearance drift with legitimate scene change, so
a controller can "win" by freezing the video. This script adds two metrics the controller
never optimized:

  motion_energy  = mean_t mean|frame_t - frame_{t-1}|            (0-255). Higher = more motion.
                   If closed-loop << open-loop here, the "win" is partly from freezing.
  tOF (warp err) = mean_t mean|warp(frame_{t-1} -> t via optical flow) - frame_t|.
                   Motion-COMPENSATED temporal inconsistency: penalizes flicker/instability
                   but NOT smooth legitimate motion. Lower = more temporally consistent.

Optical flow via cv2 Farneback (CPU, no downloads). Reads the mp4s already written by
demo_closed_loop.py and the per-item *_drift.json for the DINOv3 numbers.

A fair win for the controller = lower tOF and/or lower DINOv3 drift AT COMPARABLE motion.
Outputs fair_eval.json + fair_eval_pareto.png + a printed table.
"""

import argparse, glob, json, os
import numpy as np


def read_video_gray_and_rgb(path, resize_w=0):
    import cv2
    cap = cv2.VideoCapture(path)
    rgb, gray = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if resize_w and fr.shape[1] > resize_w:
            h = int(fr.shape[0] * resize_w / fr.shape[1])
            fr = cv2.resize(fr, (resize_w, h), interpolation=cv2.INTER_AREA)
        rgb.append(fr.astype(np.float32))                      # BGR, [H,W,3]
        gray.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return rgb, gray


def motion_energy(rgb):
    d = [np.abs(rgb[t] - rgb[t - 1]).mean() for t in range(1, len(rgb))]
    return float(np.mean(d)) if d else 0.0


def tof(rgb, gray):
    """Motion-compensated temporal warping error via Farneback flow."""
    import cv2
    H, W = gray[0].shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    errs = []
    for t in range(1, len(gray)):
        flow = cv2.calcOpticalFlowFarneback(gray[t - 1], gray[t], None,
                                             0.5, 3, 15, 3, 5, 1.2, 0)
        mapx = (gx + flow[..., 0]).astype(np.float32)
        mapy = (gy + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(rgb[t - 1], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        errs.append(np.abs(warped - rgb[t]).mean())
    return float(np.mean(errs)) if errs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--modes", nargs="+", default=["open", "pid", "pid2"])
    ap.add_argument("--resize_w", type=int, default=384, help="downsample width for flow (0=full)")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    names = sorted(os.path.basename(f).replace("_drift.json", "")
                   for f in glob.glob(os.path.join(args.folder, "*_drift.json")))
    names = names[args.shard_id::args.num_shards]
    per_mode = {m: {"motion": [], "tof": [], "drift_final": [], "drift_mean": []} for m in args.modes}
    rows = []
    for name in names:
        drift = json.load(open(os.path.join(args.folder, f"{name}_drift.json")))
        row = {"name": name}
        for m in args.modes:
            mp4 = os.path.join(args.folder, f"{name}_{m}.mp4")
            if not (os.path.exists(mp4) and m in drift):
                continue
            rgb, gray = read_video_gray_and_rgb(mp4, resize_w=args.resize_w)
            me, to = motion_energy(rgb), tof(rgb, gray)
            df, dm = float(drift[m][-1]), float(np.mean(drift[m]))
            per_mode[m]["motion"].append(me); per_mode[m]["tof"].append(to)
            per_mode[m]["drift_final"].append(df); per_mode[m]["drift_mean"].append(dm)
            row[m] = {"motion": me, "tof": to, "drift_final": df, "drift_mean": dm}
        rows.append(row)
        print(f"[{name[:28]:28}] " + "  ".join(
            f"{m}: mot={row[m]['motion']:.2f} tof={row[m]['tof']:.2f} drift={row[m]['drift_final']:.3f}"
            for m in args.modes if m in row), flush=True)

    if args.num_shards > 1:
        with open(os.path.join(args.folder, f"fair_eval_shard{args.shard_id}.json"), "w") as f:
            json.dump({"per_item": rows}, f, indent=2)
        print(f"wrote fair_eval_shard{args.shard_id}.json")
        return

    print("\n=== AGGREGATE (mean over items) ===")
    print(f"{'mode':6} {'motion':>8} {'tof':>8} {'drift_fin':>9} {'drift_mean':>10}")
    agg = {}
    for m in args.modes:
        d = per_mode[m]
        if not d["motion"]:
            continue
        agg[m] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f"{m:6} {agg[m]['motion']:8.2f} {agg[m]['tof']:8.2f} "
              f"{agg[m]['drift_final']:9.3f} {agg[m]['drift_mean']:10.3f}")

    if "open" in agg:
        base = agg["open"]
        print("\n=== vs open (%) — negative motion = more static; negative tof/drift = better ===")
        for m in args.modes:
            if m == "open" or m not in agg:
                continue
            dm = 100 * (agg[m]["motion"] - base["motion"]) / base["motion"]
            dt = 100 * (agg[m]["tof"] - base["tof"]) / base["tof"]
            dd = 100 * (agg[m]["drift_final"] - base["drift_final"]) / base["drift_final"]
            print(f"  {m:5}: motion {dm:+.1f}%   tOF {dt:+.1f}%   drift {dd:+.1f}%")

    out = {"per_item": rows, "aggregate": agg}
    with open(os.path.join(args.folder, "fair_eval.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Pareto: drift (y, lower better) vs motion (x, higher better). Good = upper-left region.
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.5, 5))
        colors = {"open": "#888", "pid": "#d62728", "pid2": "#1f77b4", "i": "#2ca02c", "p": "#ff7f0e"}
        for m in args.modes:
            if m not in agg:
                continue
            xs = per_mode[m]["motion"]; ys = per_mode[m]["drift_final"]
            plt.scatter(xs, ys, s=22, alpha=.5, color=colors.get(m, None))
            plt.scatter([agg[m]["motion"]], [agg[m]["drift_final"]], s=180, marker="*",
                        edgecolor="k", color=colors.get(m, None), label=f"{m} (mean)", zorder=5)
        plt.xlabel("motion energy  (higher = more motion preserved →)")
        plt.ylabel("DINOv3 final drift  (lower = less drift ↓)")
        plt.title("Fairness Pareto — a real win is UP-LEFT (less drift, motion kept)")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(args.folder, "fair_eval_pareto.png"), dpi=120)
        print(f"\nwrote {args.folder}/fair_eval.json and fair_eval_pareto.png")
    except Exception as e:
        print(f"(pareto plot skipped: {e})")


if __name__ == "__main__":
    main()
