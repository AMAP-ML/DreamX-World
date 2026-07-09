# Closed-Loop PID Control for DreamX-World AR Video Generation

**A test-time, training-free feedback-control layer for autoregressive world models.**

Working title: *From Open Loop to Closed Loop, in Time: Online PID Control of Autoregressive World Models*

Extends the ECCV 2026 image work ("From Open Loop to Closed Loop") from fictitious-iteration refinement of a single image to **real-time online control over the autoregressive rollout of a world model**.

---

## 0. TL;DR

DreamX-World is, at inference time, a **pure open-loop autoregressive generator**: camera/action commands and the initial image are fed strictly feed-forward, chunk after chunk, with **no runtime error measurement and no correction path back into the model**. The only "correction" present in the released code is a **post-hoc, pixel-space, single-channel (color) mean/std matcher** (`utils/postprocess.py`) applied *after* all frames exist — it cannot influence generation.

We propose to close the loop **in the temporal axis**: at each autoregressive chunk we (1) *sense* the discrepancy between the generated chunk and a reference setpoint (appearance/identity, and optionally camera trajectory), and (2) *actuate* a correction that is written **into the KV cache / context latents** so it propagates to all future chunks. The controller is a **leaky PID** (anti-windup for long rollouts), optionally 2-DOF (feedforward + PID) for the moving camera setpoint. Entirely test-time, training-free, and model-agnostic.

---

## 1. Current State of the Model — Is There Any Feedback?

**Verdict: DreamX-World is an open-loop AR generator. There is zero runtime feedback in the released inference code.** The one appearance-stabilizing mechanism that exists is open-loop and post-hoc.

### 1.1 What DreamX-World is

- **Backbone:** Wan2.2-TI2V-5B, adapted to a **causal (autoregressive) camera-controllable** world model (`wan/modules/causal_camera_model_2_2_prope_infinity.py`, class `CausalWanModel`). Distilled to **4 denoising steps** (`denoising_step_list: [1000, 750, 500, 250]`, `configs/dreamx-ar/causal_camera_forcing_5b.yaml`).
- **Chunk-wise AR rollout:** frames are generated in blocks of `num_frame_per_block: 3` latent frames. The rollout is the temporal loop in `pipeline/pipeline_causal_camera.py:154` (`for i, current_num_frames in enumerate(all_num_frames)`).
- **State carried forward:** a **KV cache** (`self.kv_cache1`, `pipeline_causal_camera.py:256`) with **sliding-window local attention** (`local_attn_size=12`, `sink_size=3`; `utils/wan_wrapper.py:102`). So the model conditions on a bounded window of its own past outputs plus a 3-frame attention sink — this is the AR "context/state."
- **Conditioning inputs (the open-loop command channels):**
  - `initial_latent` — VAE-encoded input image, seeded as frame 0 of the noise tensor (`inference_ar_forcing.py:292-297`). This is the appearance/identity anchor.
  - `y_camera` — a **PRoPE dict** `{viewmats, K}` (`cam_params_to_prope_dict`, `inference_ar_forcing.py:74`). Precomputed for the **entire trajectory up front** from `action_seq` + `action_speed_list`, then **sliced per chunk** (`pipeline_causal_camera.py:162-168`). This is the commanded camera/action signal.
  - `conditional_dict` — text-prompt embeddings (fixed for the rollout).
- **Commit-to-state step:** after each chunk's `denoised_pred` is produced, a **"rerun with context noise"** call (`pipeline_causal_camera.py:217-226`, `context_noise: 0.1`) re-encodes that chunk into the KV cache so the next chunk attends to it.

### 1.2 The generation dataflow (per chunk `i`)

```
                     ┌─────────────────── OPEN LOOP (feed-forward only) ───────────────────┐
 initial_latent ─┐   │                                                                     │
 y_camera[slice] ─┼──▶ generator (4-step denoise)  ──▶ denoised_pred  ──▶ output[chunk i]  │
 text_embeds ────┘   │        ▲                              │                             │
 KV cache (state) ───┘        │                              ▼                             │
        ▲                     │                    rerun w/ context_noise                  │
        └─────────────────────┴──────────────── writes chunk i into KV cache ─────────────┘
                                                              │
                                                              ▼   (loop to chunk i+1)
   ... after ALL chunks: single VAE decode ──▶ postprocess_video_frames (color match) ──▶ mp4
```

There is **no arrow from any measurement of the output back to the conditioning**. That missing arrow is exactly what this project adds.

### 1.3 The only existing "correction" — and why it is open-loop

`utils/postprocess.py :: postprocess_video_frames` performs **Lab-space color mean/std transfer** toward a reference frame, re-anchoring the reference at chunk boundaries (`postprocess.py:53-55`). Characterized in control terms it is:

- **Open-loop / feed-forward:** applied *after* the full video is generated and VAE-decoded (`inference_ar_forcing.py:324`); the correction can never reach the generator or KV cache.
- **Single-channel:** operates only on color statistics (L, a, b), not identity/structure/geometry.
- **P-only, static gain:** a fixed `color_correction_strength=0.3` blend — proportional, no integral or derivative, no error-adaptive gain.

**This is a gift for the paper's framing:** DreamX's own authors evidently observed appearance drift and bolted on a crude proportional pixel corrector. We *subsume and generalize* it — moving the correction (a) **inside** the loop (into the KV cache), (b) to **multiple channels** (appearance/identity + camera), and (c) to a **full PID** with anti-windup and error-adaptive gain.

### 1.4 What about the advertised "memory retrieval"?

The README advertises "geometry-guided memory retrieval" for remember-and-revisit. **This is not in the released code** — `utils/memory.py` is only GPU offload/swap plumbing (`DynamicSwapInstaller`, `get_cuda_free_memory_gb`). So we cannot rely on it, but it also means **the released model has no long-range consistency mechanism at all beyond the 12-frame local-attention window** — drift is expected and unmitigated. Good news for demonstrating a feedback effect.

### 1.5 Why drift happens here (the disturbance model)

- **Exposure bias / compounding error:** each chunk conditions on the model's own (slightly off) previous outputs via the KV cache. Errors accumulate as a slow ramp — the textbook target of **integral control**.
- **Bounded context:** with `local_attn_size=12`, the model forgets anything older than ~12 latent frames except the 3-frame sink. Appearance set at frame 0 is **not directly visible** after the window slides past it → identity/color/style drift (exactly the failure the README's "progressive training" and the postprocess hack both try to paper over).
- **Distilled few-step sampling:** 4-step denoising is lower-fidelity per chunk → larger per-step disturbance injected into the loop.

### 1.6 Consequence for the control design

- The plant is **nonlinear, stochastic, with memory (dead-time)**: a correction injected at chunk `i` influences outputs over the next ~`local_attn_size/num_frame_per_block ≈ 4` chunks as it sits in the KV window.
- **No CFG knob to modulate.** The distilled model runs a fixed 4-step schedule; the classic "CFG-scale as P-controller" actuator from the image setting is **unavailable**. We must use DreamX-native actuators (Section 2.4).

---

## 2. PID Control Design

### 2.1 Control-systems mapping

| Control concept | DreamX-World object | Code anchor |
|---|---|---|
| Discrete time step `k` | AR chunk index `i` | `pipeline_causal_camera.py:154` |
| Plant | causal generator (4-step denoise) | `WanDiffusionCameraWrapper.forward` |
| Plant state `x_k` | KV cache `self.kv_cache1` + committed `output` latents | `:256`, `:214` |
| Output `y_k` | `denoised_pred` (chunk `i`, latent) | `:214` |
| Appearance setpoint `r_app` | `initial_latent` (frame 0) | `inference_ar_forcing.py:292` |
| Camera setpoint `r_cam,k` | `y_camera` slice for chunk `i` (moving) | `:162-168` |
| Actuator `u_k` | correction to `denoised_pred` / `context_noise` / future `y_camera` | Section 2.4 |
| Disturbance `d_k` | exposure bias + 4-step sampling noise | Section 1.5 |
| Sensor | latent-moment probe or decode+extractor | Section 2.3 |

### 2.2 Regime choice — **online single-pass feedback (regime b)**

Two options existed:
- **(a) Per-chunk re-roll** (the image recipe lifted): regenerate each chunk N times, keep best. *Rejected as the primary method* — N× cost, and per-chunk cherry-picking breaks temporal continuity.
- **(b) Online single-pass:** correct chunk `i`, feed the correction **forward** into the KV cache for chunk `i+1`. **This is the method.** It is streaming-compatible, ~1× cost (plus optional one decode/chunk), and is the faithful control-theory story (a real controller over real plant dynamics).

We keep (a) as an optional "high-quality offline" ablation only.

### 2.3 Where the feedback comes from (the sensor)

Latents remain latent until a **single** VAE decode at `pipeline_causal_camera.py:242`. Sensing mid-rollout therefore trades cost vs. fidelity. Two sensors, run as an ablation axis:

**S1 — Latent-moment probe (cheap, no decode, streaming).**
Per-channel mean/std of the chunk latent `denoised_pred` → a `2C`-dim appearance vector. This is the **same statistic DreamX's postprocess uses (Lab mean/std), but in latent space**, so it captures the color/style/exposure drift the authors already care about, at ~zero cost.
`sensor(latent) = concat(mean_c, std_c)`; error `e = r_app − y`.

**S2 — Decode-and-extract (faithful, one decode/chunk).**
Decode only the chunk's **last frame** (`vae.decode_to_pixel(latent[:, -1:])`) and run a real extractor:
- **Identity present (character):** InsightFace `buffalo_l`/`antelopev2` embedding (reuse the Portrait scripts' `app.get`) → cosine error, exactly as `pid_ipa.py`.
- **Scene appearance:** DINOv2 / CLIP image embedding → cosine error.
The VAE is causal with a temporal cache (`clear_cache`), so decode the last frame with a cache-safe call; this is the main implementation subtlety of S2.

**Optional temporal sensor for the flagship (S3 — camera tracking).**
Run a visual-odometry / relative-pose estimator (or an inverse-dynamics model) on two decoded frames to get the **realized** camera motion, compare to the **commanded** `y_camera` slice → pose error. Enables closed-loop *trajectory tracking* (Section 3.2).

### 2.4 What we actuate (DreamX-native actuators)

Injected via a `chunk_callback(ctx)` fired **after** `denoised_pred` is computed (`:214`) and **before** the context-noise rerun (`:217`) — so corrections land in the KV cache.

- **(A) Context-latent state feedback (primary).** Nudge the chunk latent `denoised_pred` toward the PID target before it is committed. For S1 this is a moment-matching affine map in latent space (shift+scale per channel); for S2 it is a guided nudge along the gradient of the appearance error (or, cheaply, re-inject a blend of `initial_latent` structure). Because the "rerun with context noise" call re-encodes the *corrected* chunk into `self.kv_cache1`, the correction **propagates to all future chunks for free**. This is direct state feedback — the KV cache *is* the state.
- **(B) Adaptive `context_noise` (scalar, near-free).** `context_noise` (fixed `0.1`) controls how much the model trusts its own generated context. Drift high → **raise** it (trust the drifted context less, lean on prompt/setpoint); drift low → **lower** it. An error-adaptive gain, DreamX-native, one line.
- **(C) Future `y_camera` steering (flagship, for trajectory tracking).** The camera command for chunk `i+1…` is precomputed but **not yet consumed**. On a measured pose error we correct the upcoming `viewmats` slices so realized motion tracks the command (2-DOF: feedforward = commanded motion, PID cleans up residual).
- **(D) Periodic re-anchor (outer loop / anti-windup).** Every `reanchor_every` chunks, blend `initial_latent` back into the context and reset the integrator. Fights window-slide identity loss and doubles as anti-windup reset.

### 2.5 The controller — leaky PID (with the long-rollout fixes)

Base rule mirrors the image work (`README` update rule), then hardened for a **real** (unbounded-horizon) time axis:

```
e_k       = r − y_k                          # error = setpoint − measured
I_k       = leak · I_{k-1} + e_k             # LEAKY integral (anti-windup)  [leak<1]
I_k       = clamp(I_k, ±I_max)               # integral clamp (anti-windup)
d_k       = lowpass( −(y_k − y_{k-1}) )      # derivative ON MEASUREMENT, low-passed
u_k       = r + FF_k + kp·e_k + ki·I_k + kd·d_k
```

Differences from the image `sum_delta += delta` (`pid_ipa.py:65`) — each is a contribution, not just plumbing:

1. **Leaky integral (`leak≈0.98`) + clamp.** The image loop ran 20 iters so unbounded `sum_delta` was fine; over **hundreds of chunks it would wind up and blow the loop up.** Leak = forgetting factor; `leak=1.0` recovers the original. Sweeping `leak` is a headline ablation.
2. **Derivative on measurement + low-pass.** Per-chunk sensors are noisy; naive `kd·(delta−delta0)` amplifies noise and setpoint kicks. Derivative-on-measurement with an EMA low-pass fixes both.
3. **Feedforward term `FF_k` (2-DOF).** For the **moving** camera setpoint, pure PID lags on a ramp. Feed the commanded motion forward so PID only cleans the residual.
4. **Manifold projection (S2/identity).** InsightFace embeds live on a sphere; renormalize `u_k` (geodesic/tangent step) so corrections stay on-manifold — an issue the image scripts silently ignore.

### 2.6 Term semantics (why each PID term is physically meaningful here)

- **P** — per-chunk tracking stiffness; how hard we pull each chunk back toward the setpoint.
- **I** — **the headline.** Exposure-bias drift is a slow ramp / steady-state error; integral control is exactly the tool that eliminates it. The I-term is what flattens the drift-vs-time curve.
- **D** — temporal-flicker / jitter damping; suppresses high-frequency chunk-to-chunk appearance oscillation, and improves stability margin against the plant's dead-time.

### 2.7 Stability / practical guards

- **Dead-time from the KV window:** a correction persists ~`local_attn_size/num_frame_per_block ≈ 4` chunks → keep gains conservative or add a Smith-predictor-style lookahead. Note in analysis.
- **Contraction condition:** provide a loop-gain bound on `(kp,ki,kd)` under the sensor→generator→sensor map for stability — the theoretical backing the image paper lacked; here it is a genuine safety analysis, not decoration.
- **Actuator saturation:** clamp `|u_k − r|` so a bad sensor reading can't drive the latent off-distribution (which would corrupt the KV cache permanently).
- **Sensor dropout:** if S2's face detector misses, hold last `u`, skip the integral update that chunk.

---

## 3. Experiments & Deliverables

### 3.1 Flagship figure — drift-vs-time (appearance/identity)

Fixed seed + trajectory + prompt; **turn OFF postprocess color correction** (`--color_correction_strength 0`) so we measure the model's true drift, not the authors' post-hoc masker.
Conditions: (a) open loop (`chunk_callback=None`); (b) **I-only** (`kp=kd=0, ki>0`) to isolate drift correction; (c) full PID.
Metric: drift = `1 − cos(sensor(chunk_i), setpoint)` (S1 and S2 both).
**Expected result: (a) ramps up, (b)/(c) flatten** — the Figure 1 of the paper.

### 3.2 World-model flagship — closed-loop camera-trajectory tracking

Actuator (C) + sensor (S3). Metric: realized-vs-commanded camera pose error over the horizon (ATE/RPE from a VO estimate). Open loop drifts off the commanded path; closed loop tracks it. This is the differentiated, world-model-native result (analogue of the image `pid_pose.py`).

### 3.3 Ablations

- `leak` sweep (windup): `1.0` (image-style) → `0.9`; show `1.0` diverges on long rollouts.
- Sensor: S1 (latent proxy) vs S2 (decoded InsightFace/DINO) — cost vs fidelity ceiling.
- Actuator: (A) alone, (A)+(B), (A)+(B)+(D).
- Gain scheduling: fixed vs `‖e‖`-scaled gains.
- Horizon: 5 s (21 frames, `inference_ar_forcing.sh`) vs 60 s (the long model) — feedback effect should grow with horizon.

### 3.4 Baselines

- DreamX open loop (raw).
- DreamX + their postprocess color correction (the existing open-loop P) — show we beat their own fix.
- Per-chunk re-roll keep-best (regime a) — show online (b) matches/beats it at a fraction of the cost.

### 3.5 Metrics

Identity/appearance: cosine similarity vs frame-0 over time (mean + slope). Temporal: warping error / flicker (tOF), FVD. Camera: ATE/RPE. Cost: wall-clock and peak memory vs baseline (target: within ~1.1–2× for S1/S2 respectively).

---

## 4. Implementation Plan (minimal surgery)

1. **Controller module** (`dreamx_closed_loop.py`, drafted): `LeakyPID`, `DreamXClosedLoop` callback, `latent_appearance` (S1) + `decoded_appearance` (S2).
2. **Pipeline hook** (`pipeline_causal_camera.py`): add `chunk_callback=None` to `inference()`; insert the callback between `:214` and `:217`; route `context_noise`, `denoised_pred`, `y_camera` through `ctx`. (~6 lines; behavior byte-identical when callback is `None`.) See `dreamx_pipeline_patch.txt`.
3. **Caller** (`inference_ar_forcing.py`): construct `DreamXClosedLoop(initial_latent=…)`, pass `chunk_callback=ctrl`; dump `ctrl.drift_log`.
4. **Harness:** open-vs-closed sweep script → drift-vs-chunk plot (Fig 1); trajectory-tracking eval for §3.2.
5. **Order of work:** S1 appearance drift (§3.1) first — cheapest, clearest, needs no extractor → get Fig 1. Then S2 InsightFace for true identity. Then §3.2 camera tracking for the flagship.

### 4.1 Risks / open questions

- **Latent-moment sensor may be too weak a proxy for *identity*** (captures color/style/exposure well; face identity needs S2 decode). Mitigation: validate the cheap loop on appearance/color first, escalate to S2 if the ceiling is too low.
- **Cache-safe partial VAE decode** for S2 (temporal VAE cache) — main engineering unknown; prototype early.
- **Corrupting the KV cache:** an over-aggressive actuator commits bad latents permanently. Mitigation: actuator saturation clamp + conservative gains + re-anchor.
- **Distilled 4-step plant may be stiff/sensitive** to latent perturbations; keep `gain` small, tune on short rollouts first.

---

## 5. Positioning / Novelty

- **vs the image paper (own prior work):** fictitious iterations over one static image → **real online control over AR temporal dynamics**; unbounded-horizon anti-windup; moving setpoint + feedforward; state (KV-cache) feedback.
- **vs DreamX's postprocess:** post-hoc pixel P-controller on one channel → **in-the-loop PID** on multiple channels feeding the plant state.
- **vs training-time drift fixes** (DreamX's "progressive training on long rollouts," scheduled sampling, DFoT context noise): those are **open-loop, training-time regularizers**. Ours is **test-time, training-free, closed-loop** — orthogonal and stackable on top of any of them.
- **vs classifier-free / classifier guidance:** guidance ≈ a P-controller already; our contribution is the **I/D terms + the online-temporal, state-feedback framing**, on a distilled model where no CFG knob even exists.

---

*Companion artifacts (scratchpad): `dreamx_closed_loop.py` (controller), `dreamx_pipeline_patch.txt` (hook), `closed_loop_ar.py` (backbone-agnostic reference).*
