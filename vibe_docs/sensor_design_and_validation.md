# DreamX-World Closed-Loop PID — Sensor Design & Validation

**Status:** sensors validated on open-loop rollouts (no closed-loop code yet).
**Date:** 2026-07-13. **Gold referee:** DINOv3 ViT-B/16 (local ckpt). **Eval:** 12 scenes, horizon 63.
**Raw data:** `../sensor_validation_results/report_g*.json` · **Summary:** `../sensor_validation_results/RESULTS.md`

---

## 1. Why a sensor at all

DreamX-World generates video open-loop: image + camera commands go feed-forward, chunk
after chunk, and nothing measures the output to correct the model. Appearance/identity
**drifts** as the 12-frame attention window slides past frame 0 and exposure-bias error
accumulates. To close the loop we must first *measure* that drift per chunk — that
measurement is the **sensor**, and the whole system is only as good as it.

The danger: the cheapest place to measure (latent statistics) is the **same space the
cheapest actuator manipulates**. A controller can then drive the measured error to zero
while the real image gets worse — Goodhart's law. So before building anything, we validate
which sensor is a *trustworthy* proxy for true appearance, and at what actuation strength.

---

## 2. The sensor abstraction

Every sensor is a callable returning a **1-D vector**, and all comparison uses one metric:

```python
sensor_error(a, b) = 1 - cosine_similarity(a, b)   # 0 = identical direction; range [0, 2]
```

**Drift** is always `error(sensor(chunk_t), sensor(chunk_0))` — distance from the frame-0
anchor. Two families: CHEAP (latent, no decode — candidate in-loop signals) and GOLD
(decoded pixels, perceptual — trusted referee that the latent actuator cannot game).

---

## 3. Cheap sensors — latent space, zero decode

Input: raw latent chunk `[B, T, C=48, H=44, W=80]`. The loop already holds these, so cost ≈ 0.

### `latent_moments` → 96-dim
Per-channel **mean and std**, pooled over batch/time/height/width, concatenated:
```python
mean = x.mean(dim=(0,1,3,4))     # 48
std  = x.std(dim=(0,1,3,4))      # 48
return torch.cat([mean, std])    # 96
```
This is the **latent-space analogue of DreamX's own Lab color mean/std postprocess** — it
captures color / exposure / global-style drift, exactly what the authors already tried to
patch. **Weakness:** it is the exact quantity actuator (A) moves → collocation / Goodhart risk.

### `latent_mean` → 48-dim
Per-channel mean only. Drops all contrast/variance; a deliberately weak baseline for the
ablation (shows moments > mean).

### `latent_pooled4` → 768-dim  ★ recommended in-loop sensor
Average over frames, adaptive-avg-pool to 4×4, flatten:
```python
x = latent.mean(dim=1)            # [B,C,H,W]  average over frames
x = adaptive_avg_pool2d(x, 4)     # [B,C,4,4]  coarse 4x4 spatial grid
return x.flatten()                # 48*16 = 768
```
Keeps **coarse spatial layout** the moments discard. Two benefits: (1) best correlation with
the gold sensor, and (2) it is **not** what the moment-actuator directly manipulates → lower
Goodhart risk.

---

## 4. Gold sensors — decoded pixels, perceptual/identity space

Input: one decoded RGB frame `[3,H,W]` in `[0,1]`. Cost: one VAE decode. They live in a
space the latent moment-actuator cannot touch → **cannot be gamed** → ground truth.

### `dino3` — DINOv3 ViT-B/16 (used here)
Resize→224, ImageNet-normalize, take the **CLS token** of `last_hidden_state` → 768-dim
perceptual scene-appearance embedding. Robust to texture noise, sensitive to real
appearance/structure change. Correct referee for these landscape/scene prompts.
```python
out = model(pixel_values=x); emb = out.last_hidden_state[0, 0]   # CLS token
```

### `arcface` — InsightFace identity
Face-identity embedding; returns `None` when no face is detected (sensor dropout, handled
upstream). Not used here (no faces in the eval set); the right gold sensor for a
character/portrait world.

### `dreamsim` — learned perceptual distance
Returns a **scalar** distance to a stored reference (not an embedding). A cheap
human-perception sanity referee.

---

## 5. How the harness uses the sensors (`validate_sensors.py`)

Runs entirely on **open-loop rollouts** — no closed-loop pipeline patch required.

### Study 1 — correlation + noise floor  *(is the cheap signal trustworthy?)*
For each chunk `t = 1..T`:
- cheap-drift = `error(cheap(t), cheap(0))`
- gold-drift  = `error(dino3(decode(t)), dino3(decode(0)))`

Report **Spearman** between the two drift sequences → does the cheap number *rank-track* the
truth? Separately, rerun chunk 1 under different seeds → cheap-error between reruns = the
**noise floor**; `SNR = drift / noise_floor`.

### Study 2 — counterfactual injection  *(will correcting actually help?)* — decisive
Take the most-drifted chunk. Apply the correction the controller *would* apply — actuator
(A), `apply_moment_correction`, which shifts+scales the latent's per-channel moments a
fraction `gain` toward frame-0's moments — **decode it, re-measure the gold sensor**:
```python
improved = gold_after < gold_before
```
This closes the sense→actuate→**truth** loop without building the controller. Gold drops →
correction genuinely helps. Cheap drops but gold rises → Goodhart proven. Sweeping `gain`
gives the dose-response curve and the safe-gain ceiling.

> Design elegance: Study 2's actuator is **moment**-based while the recommended sensor is
> **pooled4** — deliberately different spaces, so a gold improvement is real signal, not the
> actuator admiring its own reflection.

---

## 6. Results (12 items, horizon 63, DINOv3 gold)

### Dose-response — the money curve
| inject gain | Spearman pooled4 | **inject→gold improve rate** | SNR pooled4 |
|---|---|---|---|
| **0.15** | 0.70 | **0.83** ✅ | 60 |
| 0.30 | 0.70 | 0.75 | 60 |
| 0.45 | 0.70 | 0.67 | 60 |
| 0.60 | 0.70 | 0.58 | 60 |
| 0.75 | 0.70 | 0.58 | 60 |
| 0.90 | 0.70 | 0.58 | 60 |
| 1.00 | 0.70 | 0.67 | 60 |

Spearman/SNR are constant across rows (measured on the same open-loop rollout; gain only
affects Study 2) — a determinism sanity check that passed.

### Findings
1. **Closed-loop will beat open-loop.** At gentle gain 0.15, correcting a drifted chunk
   toward frame 0 improves *true* DINOv3 appearance in **83%** of items — above the 0.80 bar.
2. **Hard gain ceiling.** improve-rate falls monotonically 0.83 → 0.58 as gain 0.15 → 0.60.
   Beyond ~0.3 the moment-actuator overshoots and *harms* appearance (Goodhart onset). The
   earlier "gold got worse" smoke result was a gain=1.0 saturation artifact.
3. **Signal is real.** SNR ≈ 60 (drift ÷ seed noise floor). Not measuring noise.
4. **`latent_pooled4` is the cheap sensor** (Spearman 0.70 > moments 0.67 > mean 0.60), and
   it is not collocated with the moment actuator.

### Nuances the aggregate hides (from per-item re-read of `report_g0.15.json`)
- **improve-rate is measured on the single most-drifted (last) chunk** — worst case, not an
  average. Positive effect sizes are small: Δ(gold) ≈ 0.004–0.057 on a 0–2 scale → keep gains small.
- **Spearman ~0.70 is bimodal:** strong on ~8 scenes (0.85–0.94: items 0,1,2,4,6,11) and
  blind on ~4 (`034_w.png` pooled4 **0.17**, `51_L-shape` **0.38**, `005.png` **0.53**,
  `37_异星` **0.70 / moments 0.32**). The cheap proxy is excellent on most scenes and blind
  on a few — an actionable failure mode.
- **Even at gain 0.15 the worst-drifted item Goodharts:** item 0 (`36_Tilt_Down`, gold drift
  0.96 ≈ orthogonal) — moment-error dropped 0.367→0.314 but DINOv3 error *rose* 0.960→0.979
  (`improved: false`). Once a chunk is off-manifold, matching its moments to frame 0 fakes the
  cheap number while the real image degrades.

---

## 7. Recommendation — cheapest verifiable design

- **In-loop, every chunk:** `latent_pooled4` (zero decode, SNR 60) drives a **leaky PID with
  gentle gain ≈ 0.15**. The ~1× cost path.
- **Periodic gold audit (every N chunks, or when cheap-drift crosses a threshold):** decode
  one frame → **DINOv3**. Route it specifically at (a) scene types where correlation is weak
  and (b) chunks already past a drift threshold — the two places the cheap proxy fails. This
  is the plan's S2, now justified by data.
- **Cap actuation** where the dose-response says overshoot begins: effective per-chunk
  correction ≤ ~0.3, target **0.15**. Add an actuator-saturation clamp so a bad reading can't
  shove a latent off-manifold and corrupt the KV cache permanently.

---

## 8. Reproduce
```bash
cd /home/ma-user/work/dataset/VidGen_data_obs/wt_code/DreamX-World   # working copy w/ weights + dreamx env
# Parallel 7-gain sweep on one 8-GPU H800 node (submit frontend, port 5666):
bash run_sensor_validation_parallel.sh
# Single arm:
CUDA_VISIBLE_DEVICES=0 <dreamx-python> validate_sensors.py \
  --config_path configs/dreamx-ar/causal_camera_forcing_5b.yaml \
  --model_name ./Wan2.2-TI2V-5B --transformer_path ./configs/dreamx-ar/ \
  --base_checkpoint_path ./DreamX-World-5B/model.safetensors \
  --data_path configs/dreamx/eval.json --output_folder out/ \
  --num_output_frames 63 --gold dino3 \
  --cheap latent_moments latent_mean latent_pooled4 \
  --noise_seeds 3 --chunk_relative --inject_gain 0.15 --max_items 12
```

**Code:** `sensor_zoo.py` (sensors + DINOv3 wired to local ckpt), `validate_sensors.py`
(two studies + moment actuator).
