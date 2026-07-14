# Experiment Summary — Closed-Loop Control of DreamX-World (complete campaign)

One-page record of every experiment. Full design rationale, mechanistic theory, and
conclusions: `../vibe_docs/FINAL_REPORT.md`. Chronological log: `../vibe_docs/LOOP_TRACKER.md`.
Per-run numeric data: `*_full_eval.json` in this folder (per-item, 4-axis, paired vs open).

**Protocol** (unless noted): 16 scenes, 30 s rollouts (123 latent frames / 41 chunks), seed 42,
postprocess off, paired open-vs-controlled, 4 motion-fair axes — windowed DINOv3 drift (lag8),
camera rotation via VO, motion energy, tOF — plus anchored frame-0 drift.
**WIN bar:** windowed ↓ AND rot ≥ −3% AND motion ≥ −5% AND tOF ≤ +1%, simultaneously.
**Noise floor:** ~6% (same seed, A800 vs H800 ⇒ 6% baseline shift; sub-6% single-run effects
are not interpretable — multi-seed replication required).

## Phase 0 — sensor validation (before any control)
- Cheap latent stats vs decoded DINOv3: Spearman ≈ 0.70, SNR 60–200 → drift is real & measurable.
- Counterfactual injection: gentle corrections improve gold in 83% of scenes; gains > 0.3
  overshoot (Goodhart onset). Correctly predicted the online 15 s result (81%).

## Phase 1 — actuation controllers (all fail)
| design | idea | key numbers (Δ vs open) | verdict |
|---|---|---|---|
| A moment-PID → frozen f0 | image-domain recipe transplant | motion **−33%**, tOF −32% | freeze |
| B adaptive context_noise | trust drifted context less | rot **+7.6%**, drift ~0 | neutral, motion-safe |
| C motion-preserving DC-shift | isolate mechanics from reference | rot **−12%** despite per-chunk delta preservation | froze via KV feedback |
| D gold-gated sparse shift | verified + sparse corrections | all ≈ neutral (motion −1.2%) | no effect |
| F = D + ctx-noise compensator | offset freeze with motion knob | all axes ≤0, sub-2%, 8/16 | "WIN" **died at seed 123** |
| F amplified | push the marginal win | motion −8.8%, drift +3.5% | coupling re-triggered |
| G feedforward (no KV write) | is feedback the culprit? | WORSE (motion −8.8%, drift +3.9%) | feedback not the culprit |
| H DINOv3-gradient step | correct in metric space | proxy −7% but real video +4% | proxy-Goodhart |
| I EMA moving reference | stop fighting legit change | motion +4.3% but no drift ↓, tOF +4.6% | freeze escaped, no benefit |

**Capstone statistic:** 128 (actuator,item) pairs → corr(Δdrift, Δmotion) = **+0.37**
(p≪0.001); win quadrant only 19%, scattered. The drift↔motion waterbed.

## Phase 2 — on-manifold selection (best-of-N re-roll) + N-scaling
Gold-gated per-chunk re-roll, commit best genuine sample (nothing off-manifold enters KV).
Per-chunk machinery works (re-roll wins 60–85% of gated chunks, 2–4%/chunk score gain), but:

| N | items (seeds) | pooled Δ frame-0 | paired t | Δ lag8 | Δ rot |
|---|---|---|---|---|---|
| 3 | 32 (42, 123) | +4.2% | +0.99 | +0.0015 | −6.2° |
| 5 | 16 (42) | −3.4% | −0.75 | +0.0021 | +15.6° |
| 8 | 32 (42, 123) | −2.6% | −0.59 | −0.0022 | **−19.5°** |

- N=8: **first replicated all-drift-metric reduction** (both seeds) — but benefit saturates at
  ~3% (inside noise floor, t≈0.6) and a **selection-freeze emerges** (rot −15°/−24°).
- The failure axis rotates with dose (low N pays windowed; high N pays rotation) — **no
  operating point wins all axes** ⇒ PID-scheduled budget (implemented, unit-verified,
  `PIDRerollScorer`) is bounded by this curve and was not run (argument in FINAL_REPORT §6).

## Final verdict
**NO-GO: no test-time closed-loop design beats open-loop on every axis, at any operating
point.** Five mechanisms (each ≥2 experiments): belief-state plant (corrections read as
evidence), waterbed coupling (r=+0.37, operates through actuation AND selection),
proxy-Goodhart sensor hierarchy, wrong-reference problem, chaotic trajectory variance.
Paths that could still win (untested here): revisit-trajectory tasks, in-denoising guidance,
temporally-consistent gold sensing, camera-space PID, training-time disentanglement.
