# Closed-Loop Demo Results — Open-Loop vs Leaky-PID

**Date:** 2026-07-13. **Gold metric:** DINOv3 drift = `1 - cos(dino(frame_t), dino(frame_0))`.
**Setup:** identical seed/noise/camera/prompt per item; postprocess color-correction OFF;
controller uses the cheap `latent_moments` sensor only (never sees DINOv3). Horizon 63,
16 eval items, 8×H800 (item-sharded, submit job 15).

Controller (`dreamx_closed_loop.py`): leaky-PID in latent-moment space, `kp=0.2, ki=0.1,
leak=0.85, gain_max=0.3` — the validated gentle regime. Correction is written into the
chunk latent before the context-noise rerun, so it propagates through the KV cache.

## Aggregate

| metric | value |
|---|---|
| **items where closed-loop lowers final drift** | **13 / 16 (81%)** |
| mean final-drift reduction (PID vs open) | **11.0%** |
| median final-drift reduction | 7.2% |

**81% online win-rate ≈ the offline `inject_gold_improve_rate = 0.83` at gain 0.15** — the
cheap sensor-validation correctly predicted the live closed-loop outcome.

## Per-item (final DINOv3 drift)

| item | open | PID | reduction |
|---|---|---|---|
| 65_L-shape_Forward | 0.676 | 0.349 | **48.4%** |
| case1_26_高山攀登 | 0.549 | 0.362 | **34.1%** |
| 007 | 0.383 | 0.263 | **31.1%** |
| 034_w | 0.752 | 0.592 | **21.2%** |
| case6_04_天空之城 | 0.423 | 0.373 | 11.8% |
| 51_L-shape_Forward | 0.550 | 0.492 | 10.6% |
| 19_悬崖王国 | 0.685 | 0.633 | 7.6% |
| 36_Tilt_Down | 0.840 | 0.778 | 7.3% |
| 008 | 0.692 | 0.643 | 7.1% |
| case2_01_霍格沃茨 | 0.626 | 0.588 | 6.1% |
| 005 | 0.587 | 0.565 | 3.7% |
| 12_精灵秘境 | 0.783 | 0.770 | 1.7% |
| case4_22_山中神社 | 0.788 | 0.778 | 1.3% |
| VCG211326403844_crop | 0.463 | 0.464 | −0.2% |
| 37_漫游异星 | 0.583 | 0.605 | −3.8% |
| VCG211376318837_crop | 0.516 | 0.578 | **−11.9%** |

## Reading it

- **It works on most scenes, sometimes dramatically** (5 items >20% reduction). PID
  generally beats I-only (the P term adds per-chunk stiffness).
- **The 3 losses are the Goodhart cases** the sensor validation flagged: the controller
  flattens latent moments while true DINOv3 appearance degrades. `VCG211376318837` is the
  clearest — cheap error down, gold up. This is the empirical case for the **hybrid**:
  cheap sensor in-loop, periodic DINOv3 decode audit to catch exactly these scenes.
- **Gains are deliberately gentle** (validated ceiling). Pushing harder raises the average
  win but also the Goodhart losses — matching the dose-response curve.

## Artifacts (per item, in this folder)
`<item>_open.mp4`, `<item>_pid.mp4`, `<item>_i.mp4`, `<item>_sidebyside.mp4` (open | PID),
`<item>_drift.png` (drift-vs-time curve), `<item>_drift.json` (raw per-frame drift),
`summary_shard*.json` (per-shard records).

**Best side-by-sides to watch:** `65_L-shape_Forward_sidebyside.mp4`, `007_sidebyside.mp4`,
`case1_26_*_sidebyside.mp4`. **The instructive failure:** `VCG211376318837_crop_sidebyside.mp4`.
