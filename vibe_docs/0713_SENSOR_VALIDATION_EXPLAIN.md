# Sensor Validation — DreamX-World Closed-Loop PID

## Why this experiment exists

DreamX-World generates video **open-loop**: the input image and camera commands go in
feed-forward, chunk after chunk, and nothing ever measures the output and corrects the
model. Over a long rollout the picture **drifts** away from the starting image
(color / style / identity), because of exposure bias and a 12-frame local-attention
window that eventually forgets frame 0.

The closed-loop plan fixes this with a **PID controller** that, at each AR chunk:

1. **senses** how far the chunk has drifted from the frame-0 setpoint, and
2. **actuates** a correction written into the KV cache so it propagates forward.

That controller needs to measure drift **cheaply at every chunk**. The cheapest place to
measure is **latent space** (no image decode). But there is a trap:

> The controller's *actuator* also operates in latent space. So it can drive the cheap
> error to zero **while the real picture keeps drifting** — optimizing the meter, not the
> reality. This is **Goodhart's law**.

**This experiment answers one question before any controller is built:**
*Is the cheap, in-loop signal a trustworthy proxy for real appearance drift?*
If not, no PID tuning can save it.

It runs entirely on **open-loop rollouts** — no controller required.

---

## How it works

### The setpoint and the drift

- **Setpoint** = frame 0 (the input image — the appearance / identity anchor).
- **Drift** = per-chunk error vs. that anchor, measured as
  `1 − cosine(sensor(chunk), sensor(frame0))`.

### Two families of signal

- **CHEAP** (in-loop, latent, no decode) — candidate control signals:
  - `latent_moments` — per-channel mean + std (same statistic as the postprocess color
    hack, but in latent space). **Collocated with the actuator → highest Goodhart risk.**
  - `latent_mean` — per-channel mean only (weak baseline).
  - `latent_pooled4` — spatially pooled latent; keeps coarse structure the moments throw
    away. **Not collocated with the moment actuator → lower Goodhart risk.**
- **GOLD** (referee, decoded pixels) — `DINOv3 ViT-B/16` CLS embedding on the actual
  decoded frame. Lives in a space the latent actuator **cannot directly manipulate**, so
  it **cannot be gamed**. This is the trusted stand-in for "true appearance."

### The two sweeps

**1. Sensor sweep** (`--cheap latent_moments latent_mean latent_pooled4`)
Runs all three cheap signals over the same open-loop rollout and asks: *which one best
rank-tracks the gold DINOv3 drift, and is it above the noise floor?*
→ picks the strongest, least-gameable in-loop sensor.

**2. Gain sweep — the decisive test** (`--inject_gain 0.15 … 1.0`)
This is a **dose-response curve**. For each gain `g`, take a drifted chunk, apply the
correction the controller *would* apply toward the frame-0 setpoint **at strength g**,
then **decode and re-measure the GOLD metric**:

- gold **improves** → the sense → actuate → truth chain is sound → closed-loop **will**
  help.
- cheap error drops but gold **rises** → Goodhart / actuator saturation → not usable at
  that gain.

Sweeping `g` reveals the **usable gain ceiling** — which directly sets the PID's `kp` /
`ki` scale. So this sweep is not just pass/fail; it is **controller tuning data**.

---

## The three numbers (and what passes)

| Metric | Question it answers | Target |
|---|---|---|
| **Spearman** (cheap ~ gold) | Is the cheap signal a faithful proxy for true drift? | ≥ 0.80 |
| **SNR** (drift ÷ seed-noise) | Is the drift real, or just sampling noise? | ≥ ~3 |
| **inject_gold_improve_rate** | Does correcting the cheap signal actually improve the TRUE picture? | ≥ ~0.80 |

**Decision rules:**

- Cheap sensor **usable in-loop** iff `Spearman ≥ 0.80` **and** `SNR ≥ ~3`.
- Closed-loop **beats open-loop** iff `inject_gold_improve_rate ≥ ~0.80`.
- If cheap error drops on injection but gold does **not** improve → **promote the decoded
  DINOv3 into the loop** and pay the decode cost (the plan's "S2" sensor).

---

## What it explains / findings so far

- **Drift is real, not noise.** SNR = **60–86** across cheap sensors — drift dwarfs the
  seed-to-seed noise floor. That question is closed.
- **`latent_pooled4` > `latent_moments` > `latent_mean`** for tracking the gold signal —
  as predicted, the pooled signal (not collocated with the actuator) is the best proxy.
- **Directionality holds at gentle gain.** At gain **0.3**: aggregate Spearman ≈ 0.70,
  `inject_gold_improve_rate` ≈ **0.75** — a gentle correction toward frame 0 improves true
  DINOv3 appearance in ~3 of 4 items.
- **High gain saturates.** At gain **1.0** the correction overshoots and makes gold
  *worse* — the early "gold got worse" scare was **saturation, not a broken chain**.
- **Yellow light, with a clear next step.** Most items respond well; a ~25% minority of
  scenes are weakly tracked by the cheap sensor.

---

## Implications for the controller

1. Use **`latent_pooled4`** as the cheap in-loop sensor (best tracking, least gameable).
2. Keep gains **gentle**; the dose-response curve pins the saturation ceiling that bounds
   `kp` / `ki`.
3. For the hard ~25% of scenes, use a **hybrid sensor**: cheap `latent_pooled4` by default
   plus a periodic decoded-DINOv3 check to catch cases where the cheap proxy misleads.

---

## How it was run

Parallel gain sweep on **8× H800** (one gain per GPU), **12 items**, horizon **63 latent
frames**, **3 seeds** for the noise floor, **DINOv3** gold referee.

Per-gain outputs: `sensor_validation_par_g<gain>/sensor_validation_report.json`
(the `verdict` block holds the aggregate Spearman / SNR / improve-rate).
