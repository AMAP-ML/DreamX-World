# Actuator B: Adaptive context_noise (and why closed-loop still doesn't win)

**Date:** 2026-07-13. Companion to `fair_eval_and_30s.md`.
**TL;DR:** Motivated by the brainstorm that our only tested actuator (A, moment-overwrite)
fails all three closed-loop prerequisites, we tested the untested non-destructive knob:
**adaptive `context_noise`**. On a full 16-item 30 s fair eval it is **neutral, not a win** —
the promising 3-item tune result was noise. But it produced one genuine, mechanistically
clean finding: raising `context_noise` **increases realized camera motion**, the opposite of
freezing. The verdict stands: no closed-loop variant yet beats open-loop; the evidence points
at the sensor+reference, not the actuator.

---

## 1. What we built

Two Explore agents established two facts that reshaped the plan:
- **"Pin frame-0 in the attention sink" is already implemented.** `sink_size=3` keeps frames
  0,1,2 permanently in every attention read (`wan/modules/causal_camera_model_2_2_prope_infinity.py`);
  the model already always attends to frame-0. Drift happens *despite* the anchor → re-injecting
  frame-0 is a no-op. That idea is dead.
- **`context_noise` is a per-chunk, non-destructive knob.** Read fresh each chunk at
  `pipeline/pipeline_causal_camera.py:226` (the "rerun with context noise to update KV" step).

**Actuator B** (`dreamx_closed_loop.py`, `actuator="ctxnoise"`): each chunk, measure cheap
moment-drift and set `context_noise = clamp(base + k*max(0, drift-db), base, max)` by mutating
`pipeline.args.context_noise`; return the latent UNCHANGED. No latent surgery, no KV overwrite.
`both` = B + a gentle moment nudge.

## 2. Tuning (3 items) — looked promising, was noise

ctx_max sweep {0.25, 0.40, 0.55}, ctx_k=2.0, 3 items. At ctx_max=0.4: windowed drift **−7.5%**
with camera rotation preserved (−0.7%) — the up-left Pareto move we wanted. We picked 0.4 and
scaled to 16 items. **It did not replicate.**

## 3. Full verification (16 items, 30 s) — the honest result

| metric | open | ctxnoise | both | ctx vs open | both vs open |
|---|---|---|---|---|---|
| windowed lag-8 | 0.113 | 0.119 | 0.111 | +5.0% | -2.0% |
| windowed lag-4 | 0.077 | 0.080 | 0.075 | +3.6% | -2.8% |
| frame-0 final | 0.473 | 0.468 | 0.469 | -1.1% | -0.9% |
| camera rotation | 234.2 | 252.0 | 216.4 | **+7.6%** | -7.6% |
| pixel motion | 14.39 | 14.45 | 9.63 | +0.4% | -33.0% |
| tOF | 19.27 | 19.26 | 13.19 | -0.0% | -31.6% |

win-rate (windowed lag-8, closed<open): ctxnoise 9/16, both 8/16 — coin flip.

## 4. What we learned (real, even though it's a negative)

1. **Adaptive `context_noise` does not beat open-loop.** Neutral on drift (frame-0 −1%,
   windowed +5%), win-rate ≈ 50%. The 3-item tune was small-sample noise.

2. **But it IS a real motion knob — and it corrects our earlier mechanism guess.** Raising
   `context_noise` *increases* realized camera motion (+7.6% rotation, pixel motion preserved).
   Distrusting the drifted recent context makes the model lean on the **camera command**, so it
   expresses MORE commanded motion — the **opposite of freezing**. This is a genuine,
   controllable, non-destructive lever; it just doesn't reduce appearance drift.

3. **`both` re-confirms actuator A is the culprit.** Adding B on top of the moment nudge still
   inherits A's −33% motion freeze and −7.6% rotation loss. The destructive moment-overwrite,
   not the sensor-in-general, is what tanks the fair metrics.

## 5. Conclusion & where the evidence now points

Across every actuator we have tried — moment-overwrite (A), adaptive context_noise (B), and
their combination — **no closed-loop variant beats open-loop on the fair 30 s eval.** The
instinct that feedback should help is not wrong; what's wrong is the *ingredients*:

- The **sensor** (global latent moments) is too coarse and collocated → can't isolate unwanted
  drift from legitimate change. This is now the prime suspect, not the actuator.
- The **reference** (frozen frame-0) fights legitimate scene evolution.
- Only actuator A is destructive; B is safe but, with a bad sensor/reference, has nothing good
  to actuate on.

**Next (not yet done):** the two deep fixes the data keeps pointing at —
(a) **gold-in-the-loop sensor**: decode one frame every N chunks and use DINOv3 as the error
signal (identity/style-sensitive, un-collocated); (b) **appearance/structure disentanglement**:
correct only the color/exposure sub-component, leave content/geometry (motion) free. Until the
sensor and reference are fixed, more actuators won't move the needle.

**Artifacts:** `demo_ctxnoise_30s/` (per-item `_open/_ctxnoise/_both/_sidebyside.mp4`,
`_drift.json`, `full_eval.json`); tune outputs `tune_ctx_cm{0.25,0.40,0.55}/`.
**Code:** `dreamx_closed_loop.py` (`actuator` param), `demo_closed_loop.py` (ctxnoise/both
modes, `--ctx_*`, `--sink_size`), `run_ctxnoise_30s_parallel.sh`, `run_ctxnoise_eval.sh`.
