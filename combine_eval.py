"""
combine_eval.py — combine window_eval + camera_track + fair_eval shard JSONs into one
verdict table for a demo output folder. Reusable across experiments.

Usage:
  python combine_eval.py --folder demo_shift_30s --modes open shift --baseline open
Prints per-metric open-vs-variant deltas + win counts, and writes <folder>/full_eval.json.
A variant WINS (beats open on every axis) iff: windowed lag-8 lower AND camera rotation within
-3% of open (not freezing) AND motion within -5% of open AND tOF not worse.
"""
import argparse, glob, json, os
import numpy as np


def load(folder, pat):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, pat))):
        rows += json.load(open(f)).get("per_item", [])
    return rows


def agg(rows, m, k):
    v = [r[m][k] for r in rows if m in r and k in r[m]]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--baseline", default="open")
    args = ap.parse_args()
    F = args.folder

    win = load(F, "window_eval_shard*.json")
    cam = load(F, "camera_track_shard*.json")
    fair = load(F, "fair_eval_shard*.json")
    print(f"N: window={len(win)} camera={len(cam)} fair={len(fair)}")

    METRICS = [("drift0_final", win), ("drift0_mean", win), ("lag4_mean", win),
               ("lag8_mean", win), ("rot_realized_deg", cam), ("motion", fair), ("tof", fair)]
    b = args.baseline
    hdr = f"{'metric':16} {b:>9}" + "".join(f"{m:>12}" for m in args.modes if m != b)
    print(hdr)
    table = {}
    for name, rows in METRICS:
        o = agg(rows, b, name)
        cells = []
        for m in args.modes:
            if m == b:
                continue
            c = agg(rows, m, name)
            d = 100 * (c - o) / o if o else float("nan")
            cells.append(f"{c:.3f}({d:+.1f}%)")
            table[(name, m)] = (o, c, d)
        print(f"{name:16} {o:9.3f} " + " ".join(f"{c:>11}" for c in cells))

    # verdict per variant
    print("\n=== VERDICT (beats open on every axis?) ===")
    for m in args.modes:
        if m == b:
            continue
        lag8 = table.get(("lag8_mean", m), (0, 0, 0))[2]
        rot = table.get(("rot_realized_deg", m), (0, 0, 0))[2]
        mot = table.get(("motion", m), (0, 0, 0))[2]
        tof = table.get(("tof", m), (0, 0, 0))[2]
        win_all = (lag8 < 0) and (rot > -3) and (mot > -5) and (tof <= 1)
        # per-item win rate on windowed lag8
        wr = sum(1 for r in win if m in r and r[m]["lag8_mean"] < r[b]["lag8_mean"])
        print(f"  {m}: lag8 {lag8:+.1f}%  rot {rot:+.1f}%  motion {mot:+.1f}%  tOF {tof:+.1f}%  "
              f"| win-rate(lag8) {wr}/{len(win)}  ->  {'WIN' if win_all else 'no'}")

    json.dump({"window": win, "camera": cam, "fair": fair},
              open(os.path.join(F, "full_eval.json"), "w"))
    print(f"\nwrote {F}/full_eval.json")


if __name__ == "__main__":
    main()
