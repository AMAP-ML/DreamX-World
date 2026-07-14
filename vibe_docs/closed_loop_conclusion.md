# Can Test-Time Closed-Loop PID Beat Open-Loop on Every Axis? — A Well-Evidenced Negative

**Question:** Is there a sensor+actuator making closed-loop PID control of DreamX-World's AR
rollout better than open-loop on **every** fair axis (windowed DINOv3 drift ↓, camera motion
preserved, motion energy preserved, tOF ↓), with enough items to be statistically convincing?

**Answer (8 actuators, 16 items @30s / 8 @15s, 4 motion-fair axes):** **No.** No actuator
beats open-loop on every axis. The single all-axis neutral-or-better result (F) has sub-2%
margins and a coin-flip (8/16) win-rate — within noise. A second seed confirms F's edge is
noise (WIN→no). A clean mechanistic **dichotomy** explains why, and it is the real deliverable.

---

## 1. Setup

- Model: DreamX-World-5B (Wan2.2-TI2V-5B backbone), distilled 4-step, causal camera-controllable
  AR, chunked (3 latent frames/chunk), KV cache w/ 12-frame local attention + 3-frame sink.
- Rollout: 30 s = 123 latent frames = 41 chunks (some validation at 15 s / 63 frames). seed 42
  (F also replicated at 123), postprocess OFF (measures the model's true drift).
- Controller hook: `chunk_callback` fires after each chunk's `denoised_pred`, before the
  context-noise KV rerun — corrections propagate through the KV cache (`pipeline_causal_camera.py`).
- **Fair axes** (controller never optimizes these): windowed DINOv3 lag-4/lag-8 drift
  (`window_eval.py`), realized-vs-commanded camera rotation via VO (`camera_track.py`), motion
  energy + tOF (`fair_eval.py`). A real win needs drift ↓ AND motion/camera preserved.

## 2. The scoreboard (Δ vs open-loop; WIN needs windowed↓, rot≥−3%, motion≥−5%, tOF≤+1%)

| # | actuator (sensor → actuation) | windowed lag8 | camera rot | motion | tOF | frame0 | verdict |
|---|---|---|---|---|---|---|---|
| A | latent moments → match frozen frame-0 (std+mean) | −3% | −4.5% | **−33%** | −32% | +8% | no (freeze) |
| B | drift → raise context_noise (trust drift less) | +5% | **+7.6%** | +0.4% | −0.0% | −1% | no (neutral) |
| C | latent-DC shift → frozen frame-0 (motion-preserving op) | +1.9% | **−12.2%** | −6.9% | −5.3% | +4.5% | no (froze via KV) |
| D | gold-GATED sparse DC-shift (DINOv3 confirms drift) | +1.3% | −4.8% | −1.2% | −0.4% | −0.7% | no (neutral) |
| F | D + context_noise motion-compensator | **−0.1%** | −1.5% | −2.2% | −1.4% | −1.8% | *marginal* (all ≤0, but sub-2%, 8/16) |
| — | F amplified (gain0.5, ctx_k3) | +3.5% | −7.0% | −8.8% | −7.8% | +4.2% | no (amplify overwhelms compensator) |
| G | feedforward DC-correction (never fed to KV) | +3.9% | −1.5% | −8.8% | −5.0% | +2.0% | no (worse) |
| H | DINOv3-GRADIENT step (backprop perceptual drift → latent) | +4.4% | −3.7% | +0.1% | +0.3% | +4.1% | no (proxy-Goodhart) |
| I | temporal EMA smoothing (moving reference, not frozen f0) | +0.3% | −5.7% | **+4.3%** | +4.6% | +5.2% | no (no drift ↓) |

(H, I evaluated at 15 s / 8 items; all others 30 s / 16 items.)

## 3. The dichotomy (the core finding)

Every actuator falls into exactly one bucket, **never both**:

- **(a) Anchor the OUTPUT LATENT toward a reference** (A, C, D, F, amp-F, H): reduces a drift
  proxy but **freezes motion**. The correction is committed to the KV cache and compounds over
  41 chunks, biasing the recurrent context toward a static anchor → the model *generates* less
  motion. Proven decisively by C: a per-chunk operation that provably preserves temporal deltas
  *still* froze motion (−12% rotation) via the KV feedback.
- **(b) Avoid the freeze** (B context_noise, I smooth-EMA): preserves or even **adds** motion
  (B +7.6% rot, I +4.3% motion) — but then **does not reduce** windowed/perceptual drift.

No actuator achieved both drift-reduction AND motion-preservation. Appearance-drift correction
and motion are **coupled** through the latent/KV feedback, and test-time control cannot decouple
them for this distilled AR model.



## 3b. The coupling, quantified (128 actuator×item pairs)

Pooling every (actuator, item) pair across all 9 runs — windowed-drift Δ vs camera-rotation Δ
(both % vs open), `coupling_analysis.py`, plot `coupling_frontier.png`:

- **corr(drift Δ, motion Δ) = +0.37** (N=128, p≪0.001): reducing drift systematically coincides
  with losing motion. The dichotomy is a measured trade-off, not an anecdote.
- WIN quadrant (drift↓ AND motion kept ≥ −3%): **19%** of pairs. Drift-not-reduced: **52%**.
- Among pairs that DID reduce drift (61), only **39%** kept motion.
- The ~19% wins are SCATTERED across items — no single actuator lands there consistently, so no
  actuator wins in aggregate. This is exactly why F's seed-42 "win" (a handful of lucky-item
  hits) did not reproduce at seed 123.

So: individual lucky item-actuator pairs can escape the trade-off ~1/5 of the time, but the
coupling is strong enough (r=+0.37) that no actuator does so reliably -> no robust all-axis win.

## 4. Three mechanistic barriers (each proven by ≥2 experiments)

1. **KV-feedback motion coupling** — output-latent corrections toward a static anchor freeze
   motion regardless of per-chunk mechanics (A, C, amp-F; C is the clean proof).
2. **Weak cheap target** — latent-DC / moment correction doesn't reduce *perceptual* (DINOv3)
   drift; shifting latent means ≠ reducing appearance drift (G feedforward worse, amp-F worse).
3. **Perceptual target doesn't transfer** — the DINOv3 gradient through a *single-frame* VAE
   decode (H) gamed its own decode path (−7% on that proxy) while the real full-temporal-decode
   video got worse (+4%). Classic proxy-Goodhart.

## 5. What WOULD be needed (out of test-time scope)

The barriers are structural, so the fixes are training-time / architectural, not another
test-time actuator:
- A **disentangled** representation separating appearance (correct) from motion/content (leave
  free), so correction is motion-orthogonal by construction.
- A correction signal on the **same manifold the metric lives on** (full-sequence perceptual),
  not a cheap collocated proxy or a single-frame decode.
- Breaking the **KV coupling** — e.g. correct conditioning/attention, or a training-time
  consistency loss — rather than overwriting output latents inside the recurrent loop.

## 6. Bottom line

For this distilled 4-step AR world model, **test-time closed-loop PID cannot beat open-loop on
every axis** — not for lack of trying (8 actuators spanning latent-moment, context-noise,
gold-gated, feedforward, gradient, and moving-reference designs) but because drift-correction
and motion are coupled through the KV feedback in a way no test-time latent actuator decoupled.
The honest best is **neutral** (F), within noise. This is a clean, useful negative: it says the
fix is a different *representation/training*, not a better test-time controller.

**Artifacts:** `LOOP_TRACKER.md` (full run log), per-run `demo_*/full_eval.json`,
`combine_eval.py` (verdict tool), `dreamx_closed_loop.py` (all 8 actuators),
`window_eval.py` / `camera_track.py` / `fair_eval.py` (fair metrics).
**Replication (F's only "win" was noise):** base-F re-run at seed 123 FLIPS signs vs seed 42 and
the verdict flips WIN→no:
| axis | seed 42 | seed 123 |
|---|---|---|
| windowed lag8 | −0.1% (win) | +1.1% (worse) |
| frame0 | −1.8% | +4.3% |
| camera rot | −1.5% | −2.7% |
| win-rate | 8/16 | 6/16 |
| verdict | marginal WIN | **no** |
F's apparent all-axis win does not survive a second seed → it was statistical noise. The negative
result is confirmed with replication: NO actuator beats open-loop on every axis.
