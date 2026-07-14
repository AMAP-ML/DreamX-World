"""dose_response.py — pooled N-scaling analysis for best-of-N re-roll.
Ingests each run's full_eval.json (window/camera/fair per-item lists), computes PAIRED
per-item deltas (reroll - open), aggregates per run and per N (pooling seeds), and reports
a paired t-statistic + sign test so the dose-response claim has real statistics."""
import json, os
import numpy as np

RUNS = [
    # folder, N_total, seed, hardware
    ("demo_reroll_30s",       3, 42,  "A800"),
    ("demo_reroll_s123_30s",  3, 123, "H800"),
    ("demo_reroll_n4_30s",    5, 42,  "H800"),
    ("demo_reroll_n8_s42",    8, 42,  "H800"),
    ("demo_reroll_n8_s123",   8, 123, "H800"),
]
MODE = "reroll"

def paired(run):
    f = os.path.join(run, "full_eval.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    win = {r["name"]: r for r in d["window"]}
    cam = {r["name"]: r for r in d["camera"]}
    fair = {r["name"]: r for r in d["fair"]}
    rows = []
    for n in win:
        if MODE not in win[n]:
            continue
        rows.append({
            "d_drift0": win[n][MODE]["drift0_final"] - win[n]["open"]["drift0_final"],
            "d_drift0m": win[n][MODE]["drift0_mean"] - win[n]["open"]["drift0_mean"],
            "d_lag8": win[n][MODE]["lag8_mean"] - win[n]["open"]["lag8_mean"],
            "d_rot": cam[n][MODE]["rot_realized_deg"] - cam[n]["open"]["rot_realized_deg"],
            "d_motion": fair[n][MODE]["motion"] - fair[n]["open"]["motion"],
            "o_drift0": win[n]["open"]["drift0_final"],
        })
    return rows

def t_and_sign(deltas):
    x = np.asarray(deltas, float)
    n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 1 and x.std(ddof=1) > 0 else float("nan")
    wins = int(np.sum(x < 0))
    return x.mean(), t, wins, n

print(f"{'run':22} {'N':>2} {'seed':>4} {'hw':>5} | {'d_drift0':>9} {'t':>6} {'win':>6} | {'d_lag8':>8} {'d_rot':>7} {'d_mot':>7}")
per_N = {}
for run, N, seed, hw in RUNS:
    rows = paired(run)
    if rows is None:
        print(f"{run:22} {N:>2} {seed:>4} {hw:>5} | (pending)")
        continue
    m, t, w, n = t_and_sign([r["d_drift0"] for r in rows])
    l8 = np.mean([r["d_lag8"] for r in rows]); rot = np.mean([r["d_rot"] for r in rows])
    mo = np.mean([r["d_motion"] for r in rows])
    print(f"{run:22} {N:>2} {seed:>4} {hw:>5} | {m:+9.4f} {t:+6.2f} {w:>3}/{n:<2} | {l8:+8.4f} {rot:+7.1f} {mo:+7.2f}")
    per_N.setdefault(N, []).extend(rows)

print("\n=== pooled by N (seeds combined; paired per-item deltas) ===")
print(f"{'N':>2} {'items':>5} | {'d_drift0':>9} {'t':>6} {'wins':>7} | {'d_drift0%':>9} {'d_lag8':>8} {'d_rot_deg':>9}")
for N in sorted(per_N):
    rows = per_N[N]
    m, t, w, n = t_and_sign([r["d_drift0"] for r in rows])
    base = np.mean([r["o_drift0"] for r in rows])
    l8 = np.mean([r["d_lag8"] for r in rows]); rot = np.mean([r["d_rot"] for r in rows])
    print(f"{N:>2} {n:>5} | {m:+9.4f} {t:+6.2f} {w:>3}/{n:<3} | {100*m/base:+8.1f}% {l8:+8.4f} {rot:+9.1f}")
print("\n(negative d_drift0 = reroll reduces anchored drift; t < -2 ~ p<0.05 paired; "
      "d_rot >= 0 = motion safe)")
