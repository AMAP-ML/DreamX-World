"""coupling_analysis.py — quantify the drift<->motion coupling across ALL actuators.
Pools every (run, item): windowed-drift delta vs camera-rotation delta (both % vs open).
The dichotomy predicts a positive correlation (reducing drift costs motion) and an EMPTY
win-quadrant (drift down AND motion preserved). Outputs stats + a scatter plot."""
import glob, json, os
import numpy as np

RUNS = sorted(glob.glob("demo_*/full_eval.json"))
pts = []   # (drift_delta%, motion_delta%, run, item)
for f in RUNS:
    run = os.path.basename(os.path.dirname(f))
    d = json.load(open(f))
    win = {r["name"]: r for r in d.get("window", [])}
    cam = {r["name"]: r for r in d.get("camera", [])}
    modes = [k for k in (list(win.values())[0].keys() if win else []) if k not in ("name", "open")]
    if not modes:
        continue
    m = modes[0]
    for name in win:
        if name not in cam or m not in win[name] or m not in cam[name]:
            continue
        ol, al = win[name]["open"]["lag8_mean"], win[name][m]["lag8_mean"]
        orot, arot = cam[name]["open"]["rot_realized_deg"], cam[name][m]["rot_realized_deg"]
        if ol <= 0 or orot <= 0:
            continue
        pts.append((100*(al-ol)/ol, 100*(arot-orot)/orot, run, m, name))

dd = np.array([p[0] for p in pts]); mm = np.array([p[1] for p in pts])
print(f"N = {len(pts)} (run,item) pairs across {len(set(p[2] for p in pts))} runs\n")
print(f"drift delta%:  mean {dd.mean():+.1f}  std {dd.std():.1f}")
print(f"motion delta%: mean {mm.mean():+.1f}  std {mm.std():.1f}")
r = np.corrcoef(dd, mm)[0, 1]
print(f"\ncorr(drift_delta, motion_delta) = {r:+.3f}")
print("  (positive => reducing drift (dd<0) coincides with losing motion (mm<0) = COUPLING)")
# quadrant counts. WIN = drift down (dd<0) AND motion preserved (mm >= -3)
win_q  = np.sum((dd < 0) & (mm >= -3))
drift_only = np.sum((dd < 0) & (mm < -3))
tot = len(pts)
print(f"\nWIN quadrant (drift<0 AND motion>=-3%): {win_q}/{tot} = {100*win_q/tot:.0f}%")
print(f"drift-down-but-motion-lost:             {drift_only}/{tot} = {100*drift_only/tot:.0f}%")
print(f"drift-not-reduced (dd>=0):              {np.sum(dd>=0)}/{tot} = {100*np.sum(dd>=0)/tot:.0f}%")
# of the items that DID reduce drift, how many kept motion?
red = dd < 0
if red.sum():
    print(f"\nAmong items that reduced windowed drift ({red.sum()}): "
          f"kept motion (>=-3%) in {np.sum(red & (mm>=-3))}/{red.sum()} = "
          f"{100*np.sum(red & (mm>=-3))/red.sum():.0f}%")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.axhline(0, color="#bbb", lw=1); plt.axvline(0, color="#bbb", lw=1)
    plt.axhspan(-3, mm.max()+5, xmin=0, xmax=0.5, color="#d5f5d5", alpha=.5)  # win band (left of x=0)
    plt.scatter(dd, mm, s=26, alpha=.5, color="#1f77b4", edgecolor="none")
    plt.xlabel("windowed-drift Δ vs open (%)   ← reduce drift | worse →")
    plt.ylabel("camera-rotation Δ vs open (%)   ← lose motion | keep/add →")
    plt.title(f"Drift↔Motion coupling — {len(pts)} (actuator,item) pairs\n"
              f"corr={r:+.2f}; WIN quadrant (drift↓, motion kept) = {100*win_q/tot:.0f}%")
    plt.grid(alpha=.25); plt.tight_layout()
    plt.savefig("vibe_docs/coupling_frontier.png", dpi=120)
    print("\nwrote vibe_docs/coupling_frontier.png")
except Exception as e:
    print("plot skipped:", e)
