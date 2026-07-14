# Fair Evaluation & the 30-Second Stress Test

**Date:** 2026-07-13. Companion to `sensor_design_and_validation.md`.
**TL;DR:** The 15 s "closed-loop wins" headline **does not survive** a fair, long-horizon
test. At 30 s, with metrics the controller didn't optimize, the current moment-to-frozen-
frame-0 controller is neutral-to-worse on true drift and wins short-horizon partly by
**freezing motion ~33%**. This is a real, useful negative result: it says the fix is a
different *design*, not a tuning change.

---

## 1. Why frame-0 drift is an unfair metric

`drift(t) = 1 - cos(DINOv3(frame_t), DINOv3(frame_0))` conflates two things:
- **unwanted drift** — color/identity/exposure decay (what we want to remove), and
- **wanted change** — the camera moved, so the scene *should* evolve (a world model's job).

The controller's setpoint is *frame-0 forever*, so a controller that simply makes the video
**more static** scores "less drift" while being worse. Metric and controller share the same
flawed anchor, so they flatter each other. (It is not a pure tautology: the controller acts
on 96-d latent moments, the metric is DINOv3 pixels — different spaces — which is why it can
and does fail on some items.)

## 2. Fair metrics (controller never optimized these)

- **motion_energy** = `mean_t mean|frame_t − frame_{t−1}|`. Higher = more motion. If closed-
  loop ≪ open-loop, the "win" is partly freezing.
- **tOF** = motion-compensated warping error: warp `frame_{t−1}→t` via optical flow
  (cv2 Farneback), measure residual. Meant to isolate *flicker* from *legitimate motion*.
  Caveat below.

Code: `fair_eval.py` (downsampled flow, 8-way shardable). Plot: `demo_closed_loop_30s/fair_eval_pareto.png`.

## 3. Results — 15 s vs 30 s

### 15 s (horizon 63) — the original headline
Closed-loop lowered frame-0 final drift in **13/16** items (mean −11%) — **but** motion fell
**−31%** and the correlation between drift-reduction and motion-loss was ~0 (0.12), i.e. the
wins weren't *only* freezing, but freezing was a large uncontrolled confound.

### 30 s (horizon 123, 41 chunks) — the fair, stressed test
| mode | motion | tOF | drift_final | drift_mean |
|---|---|---|---|---|
| open | 14.39 | 19.27 | 0.610 | 0.517 |
| pid (v1) | 9.69 | 13.15 | 0.615 | 0.523 |
| pid2 (v2) | 9.68 | 13.16 | 0.631 | 0.511 |

**vs open:**
| | motion | tOF | drift_final | drift_mean | drift<open |
|---|---|---|---|---|---|
| pid | −32.7% | −31.8% | **+0.9%** | +1.1% | 7/16 |
| pid2 | −32.7% | −31.7% | **+3.5%** | −1.2% | 7/16 |

## 4. What this means (honest)

1. **The 15 s win does not generalize.** At 30 s, frame-0 final drift is **neutral-to-worse**
   (7/16 ≈ coin flip). Individual regressions are large (e.g. `36_Tilt_Down` open 0.766 →
   pid **0.975**). Over 41 chunks the integrator winds up and keeps forcing global moments
   back to a *frozen* frame-0 even as the camera legitimately moves the scene away — so it
   fights content and degrades frames (the plan's "KV-cache corruption" risk, observed).

2. **It still freezes motion ~33%** — the confound you flagged, now measured directly and
   unchanged at long horizon.

3. **tOF drops ~32%, but ≈ proportional to the motion drop** — so tOF as computed is *also*
   confounded by freezing (less motion ⇒ easier to warp ⇒ lower residual). It is **not**
   clean evidence of genuine flicker reduction. A cleaner version needs higher-quality flow
   (RAFT) and per-unit-motion normalization, or camera-trajectory tracking.

4. **v2 (deadband + I-dominant) barely differs from v1 at 30 s.** The deadband (0.10) is too
   small for a long horizon: drift exceeds it early and stays above, so the gate is on almost
   always and pid2 ≈ pid. v2's design intent is right; its threshold is horizon-mismatched.

## 5. Why, and what actually fixes it

The root cause is the **setpoint**, not the gains: *frame-0 forever* is wrong once the scene
legitimately evolves. Tuning kp/ki/deadband cannot fix a wrong reference. The design changes
that should:

- **Moving / leaky setpoint** — track a slow reference (EMA of recent appearance, or frame-0
  within a bounded band) so legitimate drift is allowed and only *runaway* is corrected. This
  is the direct fix for both the freezing and the long-horizon windup.
- **Decode-gold-in-the-loop** — the cheap global-moment sensor is too weak/collocated at long
  horizon; use a periodic DINOv3 decode (the plan's S2) as the in-loop signal when cheap drift
  is high.
- **Appearance/structure disentanglement** — correct only the color/exposure component, leave
  the spatial/structural component (which carries motion) free. Global per-channel moments
  couple to motion; that coupling is why we damp motion.
- **Fair metric upgrade** — RAFT-based tOF normalized per unit motion, plus camera-trajectory
  tracking (realized-vs-commanded pose), to score consistency independent of legitimate motion.

## 6. Bottom line

Pushing to 30 s with motion-aware metrics did its job: it showed the current controller is a
**dead end as designed** — it wins short clips partly by freezing and does not hold up over
long horizons. That is the honest state. The value is a sharp, evidence-backed spec for the
next controller (moving setpoint + gold-in-loop + appearance-only correction), rather than a
premature "it works."

**Artifacts:** `demo_closed_loop_30s/` — per-item `_open/_pid/_pid2/_sidebyside.mp4`,
`_drift.png`, `_drift.json`; `fair_eval.json`; `fair_eval_pareto.png`.
**Code:** `dreamx_closed_loop.py` (v2 controller w/ deadband+gating), `demo_closed_loop.py`,
`fair_eval.py`, `run_demo_30s_parallel.sh`.

---

## 7. Shifting-window drift (addendum)

Replacing the fixed frame-0 anchor with a SHIFTING reference (`window_eval.py`):
`driftLag_L[t]=1-cos(e_t, e_{t-L})` and `driftWmean_L[t]=1-cos(e_t, mean(e_{t-L..t-1}))`.
Removes the cumulative-legitimate-motion confound (only sees change over L steps).

16 items, 30s, DINOv3 re-embedded from the mp4s:

| metric | open | pid (v1) | pid2 (v2) | pid vs open | pid2 vs open |
|---|---|---|---|---|---|
| frame-0 final | 0.473 | 0.497 | 0.510 | +5.1% | +7.8% |
| lag-4 (~0.5s) | 0.077 | 0.085 | 0.075 | +9.6% | -2.7% |
| lag-8 (~1s)   | 0.113 | 0.123 | 0.110 | +8.5% | -2.8% |

**What it revealed (that frame-0 hid):** the windowed metric SEPARATES v1 and v2.
- v1 (`pid`) is *worse* on local consistency (+9-10%): its aggressive per-chunk P correction
  yanks each chunk toward frame-0 → chunk-boundary appearance jumps → local instability.
- v2 (`pid2`) is marginally *better* (-2.7%): deadband + I-dominant avoids the jumps.
  → controller lesson: the P term is harmful to temporal consistency; v2's direction is right.

**Limits (unchanged verdict):** (1) v2's win is tiny and near coin-flip (8/16). (2) Windowed
drift is STILL minimized by freezing, and v2 froze motion ~33%, so the -2.7% is not clean.
(3) Frame-0 cumulative drift is still worse for both. Conclusion: shifting window is the
better, more discriminative appearance-consistency metric, but must be read WITH motion; it
confirms closed-loop does not cleanly beat open-loop at 30s.

**Artifacts:** `demo_closed_loop_30s/window_eval.json`; code `window_eval.py`,
`run_window_eval_parallel.sh`.

---

## 8. Camera-trajectory tracking (the freeze-proof axis)

Commanded camera motion from the action trajectory; realized motion estimated by monocular
VO (ORB + essential matrix + recoverPose, `camera_track.py`) on the generated frames. A frozen
video cannot fake this: no realized rotation -> fails. Metric = summed |relative rotation|.

16 items, 30s:

| mode | realized rotation | vs open | track ratio | realized<open |
|---|---|---|---|---|
| open | 234.2 deg | -- | 0.46 | -- |
| pid  | 225.0 deg | -3.9% | 0.45 | 10/16 |
| pid2 | 223.7 deg | -4.5% | 0.44 | 11/16 |

(commanded mean 530 deg; VO fail-rate ~1%)

**This CORRECTS the "wins by freezing" conclusion.** Pixel motion-energy said -33%; camera
rotation says only **-4%**. So the camera trajectory is ~96% preserved — the closed loop is
NOT freezing the camera. Almost all of the 33% pixel-motion drop was **appearance/color
flicker** (which the controller is *supposed* to damp), not camera motion. The motion-energy
metric conflated flicker with camera motion the same way frame-0 drift conflated drift with
legitimate change; camera tracking disentangles them.

### Corrected synthesis (all four axes)
| axis | open -> closed | reading |
|---|---|---|
| frame-0 drift | worse (+5..8%) | no cumulative-drift benefit |
| windowed drift | v1 +9% / v2 -3% | v2 slightly steadier locally |
| pixel motion | -33% | mostly flicker removed |
| camera rotation | -4% | camera motion preserved |

**Updated verdict: closed-loop is roughly NEUTRAL, not a freezing cheat.** It strips some
high-frequency appearance flicker while keeping the camera trajectory essentially intact, but
does not clearly reduce cumulative appearance drift; only v2 gains a small local-consistency
edge. No clear win, but not the failure implied by pixel motion alone.

**Caveats:** VO measures rotation (scale-free, robust); monocular translation is scale-
ambiguous, so forward-dolly suppression is not directly ruled out (the -33%/-4% gap is
"flicker and/or translation"). Absolute ratio ~0.46 is VO underestimation; only relative
open-vs-closed is trustworthy.

**Artifacts:** `demo_closed_loop_30s/camera_track.json`; code `camera_track.py`.
