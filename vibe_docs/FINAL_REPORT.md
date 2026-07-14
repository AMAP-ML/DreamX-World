# Closed-Loop Control of Autoregressive Video World Models: The Complete Investigation

**Status:** COMPLETE (2026-07-14). The N-scaling study was the final experiment.
**TL;DR:** across **10 test-time controller designs** (8 actuation-based, feedforward, and
best-of-N selection at doses N∈{3,5,8}), each evaluated on 16 scenes × 30 s × 4 motion-fair
metrics with seed/hardware replication, **no closed-loop design beats open-loop on every
axis, at any operating point**. The final measurement: selection's drift benefit is real in
direction but saturates at ~3% (inside the ~6% chaos floor, t≈0.6), while a motion cost
emerges at high selection pressure (camera rotation −20° at N=8) — the same drift↔motion
waterbed (r=+0.37) that defeated every actuator, now demonstrated through the selection
channel too. The investigation's product is a mechanistic theory of *why*: a belief-state
plant that reinterprets corrections as evidence, a measured objective-level coupling, a
proxy-Goodhart sensor hierarchy, a wrong-reference problem, and chaotic trajectory variance.
Full design, results, theory, and the conditions under which closed-loop could still win are
below.

---

## 1. Motivation: why closed-loop PID for AR video generation

DreamX-World (Wan2.2-TI2V-5B backbone, distilled to 4 denoising steps, causal camera-controlled
AR generation in 3-latent-frame chunks) is, at inference, a **pure open-loop generator**:
image + camera commands enter feed-forward; nothing ever measures the output and corrects the
process. Long rollouts drift — color, exposure, style, and identity decay over time.

The control-theoretic reading is exact, not metaphorical:

| Control concept | DreamX object |
|---|---|
| Discrete time step k | AR chunk index |
| Plant | the 4-step causal denoiser |
| Plant state x_k | KV cache (12-frame sliding window + 3-frame pinned sink) |
| Output y_k | denoised chunk latent |
| Disturbance d_k | sampling noise + exposure bias |
| Setpoint r | frame-0 appearance (and the commanded camera path) |

Exposure bias — the model conditioning on its own slightly-off outputs — produces a **ramp
disturbance**, and the textbook answer to a ramp is **integral control**. Moreover the model's
own authors evidently saw the drift: the released code ships a post-hoc Lab-space color
mean/std matcher (`utils/postprocess.py`) — literally a proportional controller, applied
open-loop after generation. The hypothesis "close the loop properly and beat open-loop" was
therefore well-founded. This document records what happened when we tested it exhaustively.

## 2. The plant: what we established about the system under control

Facts established by code analysis (two independent code-exploration passes) and experiments:

- **Chunked AR with KV feedback.** Each chunk is denoised in 4 steps, then re-encoded into the
  KV cache at `context_noise=0.1` ("rerun with context noise",
  `pipeline_causal_camera.py:~226`). Future chunks attend to this cache. Any modification to a
  committed chunk therefore **propagates through all future generation** — the central
  double-edged sword of this investigation.
- **The anchor already exists.** `sink_size=3` pins frames 0–2 permanently into every attention
  read (structurally exempt from sliding-window eviction). The model *always sees frame-0*;
  drift happens despite a permanent anchor. "Add memory of the anchor" is therefore not the
  missing ingredient (and re-injecting frame-0 into its own KV slot is a literal no-op).
- **The plant is chaotic.** Identical seed + code on A800 vs H800 GPUs produces open-loop
  rollouts whose final drift differs by ~6% — bf16 numerics amplified through 41 chunks of
  recursion. **This sets the noise floor**: single-run effects below ~6% are not
  interpretable; only multi-seed paired statistics count. (Several of our own early "wins"
  died by exactly this sword; we adopted seed replication as standard after the first.)
- **No usable CFG knob.** The distilled 4-step model has no classifier-free-guidance scale to
  modulate — the classic "guidance-as-P-controller" actuator from image work is unavailable.

## 3. Experimental infrastructure (how the experiments are designed)

### 3.1 The intervention point
A minimal pipeline hook (`chunk_callback` / `chunk_scorer` in
`pipeline/pipeline_causal_camera.py`) fires after each chunk's `denoised_pred` and before the
context-noise KV commit. Behavior is **byte-identical when hooks are None** (verified: a
post-refactor open rollout reproduced a pre-refactor drift curve with max|diff| = 0.0).
`chunk_callback` supports actuation (return a modified latent); `chunk_scorer` supports
selection (re-run the chunk's 4-step denoise with fresh noise via a closure; KV bookkeeping is
safe because re-denoising at the same position takes the in-place recompute path).

### 3.2 Sensors (validated before any control was built)
`validate_sensors.py` ran two studies on open-loop rollouts *before* closing any loop:
- **Study 1 (trust):** does a cheap in-loop signal rank-track a gold perceptual signal?
  Cheap latent stats (channel moments / pooled grids) reached Spearman ≈ 0.70 vs decoded
  DINOv3 — usable but imperfect; SNR vs seed noise ≈ 60–200 (drift is real, hugely
  super-noise).
- **Study 2 (counterfactual injection — the decisive design):** apply the correction the
  controller *would* apply, decode, and check whether the **gold** metric improves. This
  predicted closed-loop improvement at 83% of scenes for gentle gains — and correctly
  identified the Goodhart failure mode (cheap error drops while gold rises) and a gain
  ceiling (~0.3) before any controller existed.
Gold sensor: DINOv3 ViT-B/16 CLS embeddings (local checkpoint); scene-appearance-sensitive,
not collocated with any latent actuator.

### 3.3 Metrics: the fairness evolution (a result in itself)
The evaluation had to be redesigned twice, because naive metrics *reward degenerate control*:
1. **Frame-0 DINOv3 drift** (1 − cos vs frame 0): conflates unwanted drift with legitimate
   scene change — a frozen video scores perfectly. Rewarded the controllers that froze motion.
2. **+ Motion energy & tOF** (optical-flow-compensated warping error): exposed a −33% motion
   freeze that frame-0 drift had scored as a "win"; but both are still minimized by freezing.
3. **+ Windowed DINOv3 drift** (lag-4/lag-8 vs a *shifting* reference): removes the
   cumulative-legitimate-change confound; more discriminative (separated controller variants
   frame-0 could not).
4. **+ Camera-trajectory tracking** (monocular VO: realized vs commanded rotation): the one
   axis a frozen video cannot fake; it *corrected* an over-harsh "freezing" verdict by showing
   camera motion ~96% preserved when pixel-motion loss was mostly flicker removal.
**Final standard: a real win = windowed drift ↓ AND camera rotation preserved AND motion
energy preserved AND tOF not worse — simultaneously.** All controllers are blind to all four.

### 3.4 Protocol
16 eval scenes (landscape/architecture, always-departing camera trajectories), 30 s rollouts
(123 latent frames, 41 chunks), fixed seed per run, postprocess color-correction OFF, paired
open-vs-controlled per item, 4-axis eval via `window_eval.py` / `camera_track.py` /
`fair_eval.py`, verdict via `combine_eval.py`, dose-response statistics via
`dose_response.py` (paired t + sign tests). Compute: 8×H800 nodes (submit API) + 4×A800 local,
item-sharded.

## 4. The campaign: every controller, its design logic, and what it taught

| # | design (sensor → actuator) | rationale | 4-axis result (Δ vs open) | lesson |
|---|---|---|---|---|
| A | latent moments → match frozen frame-0 (mean+std affine), leaky-PID | direct transplant of the image-domain recipe | drift proxy ↓ but **motion −33%**, tOF −32% | anchoring the output to a static reference freezes the world |
| B | drift → raise `context_noise` (trust drifted context less) | native, non-destructive conditioning knob | neutral drift; **rot +7.6%**, motion +0.4% | conditioning-level control is motion-safe but can't reduce drift |
| C | motion-preserving DC-shift (per-chunk deltas provably preserved) → frozen frame-0 | isolate actuator mechanics from the reference | **still froze** (rot −12.2%) | the freeze is a *KV-feedback* effect, not actuator mechanics — corrections compound through the recurrent state |
| D | gold-gated sparse DC-shift (DINOv3 confirms real drift; correct ~⅓ of chunks) | verified + sparse ⇒ minimal KV contamination | all axes ≈ neutral (motion −1.2%) | sparse enough not to freeze ⇒ too sparse to correct: dose-coupling |
| F | D + context_noise as **motion compensator** | attack the coupling with an independent motion knob | all axes ≤ 0 — formal "WIN" — but sub-2%, 8/16 | **did not replicate at seed 123 (WIN→no)**: taught us the chaos floor |
| F+ | F amplified (gain 0.5, ctx_k 3) | push the marginal win | worse everywhere (motion −8.8%) | amplification re-triggers the coupling; gentle was already optimal |
| G | **feedforward** post-hoc correction (never enters KV) | isolate feedback as the culprit | *worse* than closed-loop (motion −8.8%, drift +3.9%) | **feedback was NOT the culprit** — the latent-DC target itself is too weak |
| H | DINOv3-**gradient** step (backprop perceptual drift through VAE+DINOv3 to the latent) | correct in the metric's own space | proxy −7% but real video +4% | **proxy-Goodhart**: single-frame decode ≠ temporal decode; the gradient games its own sensor |
| I | temporal EMA smoothing (moving reference, not frame-0) | stop fighting legitimate change | motion +4.3% (freeze escaped!) but no drift reduction, tOF +4.6% | moving reference solves the freeze but latent-mean smoothing can't touch perceptual flicker |
| J | **best-of-N selection** (gold-gated re-roll, commit best genuine sample) | on-manifold by construction — the oracle for the sensor chain | see §5 | selection is motion-safe; benefit is dose-dependent and at/below the chaos floor at N≤5 |

**The quantified coupling (capstone statistic of the actuation phase):** pooling 128
(actuator, item) pairs across 9 runs: **corr(Δdrift, Δmotion) = +0.37** (p≪0.001). Reducing
drift systematically costs motion; only 19% of pairs land in the win quadrant, scattered —
no actuator lands there consistently. Plot: `coupling_frontier.png`.

## 5. The selection phase (best-of-N) and the N-scaling study

Design: at each chunk, decode the last frame, score gold drift vs frame-0; if above threshold,
re-run the chunk's denoise with fresh noise up to N−1 times and commit the best-scoring
**genuine model sample**. Nothing off-manifold ever enters the KV. Per-chunk diagnostics show
the machinery works: re-rolls beat the first sample in 60–85% of gated chunks, gaining 2–4%
score per chunk.

Replicates (paired per-item, real-video metrics):

| run | N | seed | HW | Δ frame-0 final | Δ windowed lag8 | Δ camera rot |
|---|---|---|---|---|---|---|
| 1 | 3 | 42 | A800 | +8.1% | +0.9% | −9.9° |
| 2 | 3 | 123 | H800 | +0.4% | +1.8% | −2.6° |
| 3 | 5 | 42 | H800 | −3.4% | +1.9% | **+15.6°** |
| 4 | 8 | 42 | H800 | **−1.5%** | **−2.2%** | −15.1° |
| 5 | 8 | 123 | H800 | **−3.7%** | **−1.7%** | −23.9° |

**Pooled dose-response (the campaign's final measurement):**

| N | items | pooled Δ drift₀ | paired t | wins | Δ lag8 | Δ rot |
|---|---|---|---|---|---|---|
| 3 | 32 | +4.2% | +0.99 | 13/32 | +0.0015 | −6.2° |
| 5 | 16 | −3.4% | −0.75 | 8/16 | +0.0021 | +15.6° |
| 8 | 32 | **−2.6%** | **−0.59** | 16/32 | **−0.0022** | **−19.5°** |

**Verdict against the pre-registered bar (≥10% pooled reduction, t≥2): decisively missed.**
Three findings close the question:
1. **The drift benefit is real in direction but saturates at ~3%** — inside the ~6% chaos
   floor, t≈0.6, win-rate 50%. At N=8, for the first time, *every* drift metric (frame-0 and
   windowed, both seeds) moves down — direction is replicated — but the magnitude has already
   plateaued between N=5 and N=8.
2. **The selection-freeze emerges at high dose.** Camera rotation, safe at N≤5 (even +15.6° at
   N=5), drops −15°/−24° in both N=8 arms. Under strong selection pressure, "least drifted
   candidate" systematically means "slightly more static candidate" — the waterbed coupling
   reasserts itself **through the selection channel**, exactly as it did through every
   actuation channel.
3. **No operating point wins all axes.** Low N pays windowed drift; high N pays camera motion;
   no N clears everything. The dose curve's failure axis *rotates* with N but never vanishes.

Two lessons banked along the way: (i) a "greedy selection actively harms" narrative from
replicate 1 alone did not survive replication — single-run stories are worthless on this
plant; (ii) the scorer's single-frame-decode proxy inflates its own gains ~2× vs the true
temporal decode (H's lesson, quantified again in selection).

## 6. Where PID actually belongs: PI-scheduled test-time compute — and why it was not run

Everything above relocates PID from the actuation layer (where every variant failed) to the
**supervisory layer**: let the control law allocate the correction *budget*.

```
e_k      = gold drift of chunk k's first sample
excess_k = max(0, e_k − deadband)              # tolerate legitimate change
I_k      = leak·I_{k−1} + excess_k             # leaky integral (anti-windup)
n_k      = clamp(round(kp·excess_k + ki·I_k), 0, N_max)   # candidates for chunk k
```

`PIDRerollScorer` is implemented and unit-verified (allocation ramps 0→4 as drift accumulates,
stays 0 inside the deadband, drains after successful corrections). **We did not run the
matched-budget experiment, for a principled reason:** PID scheduling *interpolates the dose*.
Its achievable outcomes are bounded by the fixed-N dose curve — and the completed N-scaling
study shows every point on that curve fails at least one axis (low N pays windowed drift,
high N pays camera motion), with the drift benefit saturating at ~3%, inside the chaos floor.
An allocator cannot win where no allocation wins. Running it would have confirmed a
mathematically bounded negative at the cost of two more node-hours; we record the argument
instead. (If a future setting moves any dose point into the win quadrant — see §8 — the
PID-scheduled version is the first thing to run, and the code is ready.)

## 7. Why closed-loop control fails here: the mechanistic theory

Five interlocking mechanisms, each established by ≥2 independent experiments:

1. **The belief-state plant.** The KV cache is not passive state — it is the model's *evidence
   about the world*. Writing a corrected latent into it doesn't apply a restoring force; it
   injects **false evidence** ("the world still looks like frame-0"), which the model
   rationally continues — as a *static* world. Proven cleanly by C: an operation that provably
   preserves per-chunk motion still froze the video *through* the KV feedback. Controlling
   this plant is closer to managing expectations in an economy than torquing a motor.
2. **The waterbed coupling.** Δdrift and Δmotion correlate at +0.37 across every actuation
   design — push drift down, motion pops out, like Bode's sensitivity waterbed. The coupling
   lives in the *objective* (proximity to a static anchor), not in any particular actuator:
   it defeated latent surgery at every gain, and the completed N-scaling study shows it
   operating **through pure on-manifold selection** as well — camera rotation safe at N≤5,
   −15°/−24° at N=8 in both seeds. Any channel that prefers anchor-proximity hard enough
   pays in motion.
3. **The proxy-Goodhart hierarchy.** Every affordable sensor is a proxy, and every controller
   that optimized its proxy hard enough broke it: latent moments (gamed by moment-matching),
   single-frame VAE decode (gamed by the gradient actuator, inflated selection's apparent
   gains 2×). The true objective — full-temporal-decode perceptual consistency — is only
   measurable offline.
4. **The reference problem.** Frame-0-forever is the wrong setpoint for always-departing
   trajectories: the controller must fight *legitimate* change to reduce *measured* drift.
   Moving references (I) fix the freeze but dilute the correction signal. (Untested fair
   arena: revisit/orbit trajectories, where frame-0 IS the right setpoint at revisit time.)
5. **Chaos.** 41 chunks of recursion amplify bf16 rounding into ~6% outcome shifts. Any
   controller effect smaller than that is invisible without multi-seed paired statistics —
   and every affordable gentle controller lives exactly in that band.

In one sentence: **the plant reinterprets control inputs as evidence, couples the correction
axis to the content axis through its own feedback, defeats every affordable sensor by proxy
gaming, and hides sub-6% improvements under chaotic variance — so test-time closed-loop
control, in every actuation form we could devise, cannot demonstrably beat open loop on all
axes at once.**

## 8. What WOULD let closed-loop win (ranked, honest)

1. **PID-scheduled selection at higher N** (⏳ this is being tested right now) — the only
   direction with a positive, motion-safe, dose-dependent signal.
2. **A fair task**: revisit/orbit trajectories (DreamX's own advertised "remember-and-revisit"
   challenge) where the anchor is legitimate and the reference conflict vanishes.
3. **In-denoising guidance**: inject the correction during the 4 denoising steps so the
   denoiser filters it (on-manifold actuation) — the principled fix for mechanism 1; untested.
4. **A temporally-consistent gold sensor** (score on context-decoded frames) — the fix for
   mechanism 3; would also recover the ~2× proxy loss in selection.
5. **Camera-trajectory PID** (plan §3.2): sensor (VO), actuator (camera conditioning), and
   metric live in the same space; zero reference conflict; still untested.
6. **Training-time**: disentangled appearance/motion representations, or a consistency loss —
   out of test-time scope but where mechanisms 1+2 ultimately point.

## 9. Artifact index

Code: `dreamx_closed_loop.py` (all controllers + scorers), `demo_closed_loop.py` (paired
generation harness), `pipeline/pipeline_causal_camera.py` (hooks; regression-verified),
`sensor_zoo.py` / `validate_sensors.py` (sensors + pre-validation),
`window_eval.py` / `camera_track.py` / `fair_eval.py` / `combine_eval.py` /
`coupling_analysis.py` / `dose_response.py` (evaluation & statistics).
Results: `demo_*/full_eval.json` per run; `coupling_frontier.png`;
`vibe_docs/LOOP_TRACKER.md` (chronological log); companion docs
(`sensor_design_and_validation.md`, `fair_eval_and_30s.md`, `actuator_b_context_noise.md`,
`closed_loop_conclusion.md`).

⏳ → **Final statement (2026-07-14): NO-GO for test-time closed-loop control of this plant.**
The dose-response study completed the picture: the last live channel (on-manifold selection)
shows a real but saturating ~3% drift benefit that never clears the chaos floor, and inherits
the motion cost at high dose. Every channel — actuation, feedforward, gradient, selection —
ultimately pays the same waterbed. The win conditions in §8 (revisit-trajectory tasks,
in-denoising guidance, temporally-consistent sensing, camera-space PID, training-time
disentanglement) are the documented paths forward; the code, metrics, and statistics built
here transfer to all of them.
