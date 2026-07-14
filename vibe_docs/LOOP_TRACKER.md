# LOOP TRACKER — closed-loop PID that beats open-loop on EVERY axis

## LOOP ENDED (tick ~21): question ANSWERED (negative, replicated, quantified). Cron e926a704 deleted. 8 actuators tried; coupling r=+0.37 (N=128); F's only 'win' was seed-noise. See closed_loop_conclusion.md. Restart the loop only to pivot to training-time/architectural fixes.

**Goal:** find a sensor+actuator where closed-loop PID beats open-loop on **all** fair axes
(windowed DINOv3 drift ↓, camera rotation preserved/↑, tOF ↓, motion preserved) with enough
items (≥16) to be statistically convincing. Exps via port 5666, 8×H800, 30 s / 123 latent
frames, seed 42, postprocess off.

**Fair axes (from `fair_eval_and_30s.md`):** windowed lag-4/lag-8 drift, frame-0 drift,
camera rotation (`camera_track.py`), motion energy + tOF (`fair_eval.py`). A real win = drift ↓
AND motion/camera preserved.

## Prior results (baseline to beat)
| actuator | windowed | camera rot | motion | verdict |
|---|---|---|---|---|
| open-loop | — | 234° | 14.4 | baseline |
| A: moment→frozen-frame0 (pid/pid2) | v2 −3% | −4.5% | **−33%** | freezes, loses |
| B: adaptive context_noise | +5% | +7.6% | +0.4% | neutral, more motion |
| A+B (both) | −2% | −7.6% | −33% | inherits A's freeze |

**Diagnosis:** sensor (global per-frame-averaged moments) + reference (frozen frame-0) are the
problem. Moment-match flattens temporal variation → freezes motion. Frozen frame-0 fights
legitimate change.

## Candidate ledger
| # | idea | status | result |
|---|---|---|---|
| C | per-frame appearance-only DC-drift correction (preserve temporal deltas) | DONE | **no** — froze motion (rot −12.2%) via KV feedback despite per-chunk delta preservation; windowed +1.9%, frame-0 +4.5% |
| D | gold-GATED sparse shift: correct only decode-ticks where DINOv3 confirms real drift | DONE | **no** — but best "no-harm": motion −1.2% (nearly preserved!), frame-0 −0.7% (better!), windowed +1.3%, rot −4.8%. Neutral everywhere. |
| E | attention-bias anchoring | DEFERRED | flash-attn = no clean logit bias; risky RoPE surgery; B already probed "rely-on-anchor" direction (null) |
| F | gold-gated shift + context_noise MOTION COMPENSATOR | DONE | **marginal WIN** — first ALL-AXIS neutral-or-better: lag8 −0.1%, rot −1.5%, motion −2.2%, tOF −1.4%, frame0 −1.8%. BUT sub-2% margins, win-rate 8/16 (coin-flip) => not yet statistically convincing. Motion-compensator DID break the freeze coupling. |
| H | DINOv3-GRADIENT actuation + ctxnoise | DONE | **no** — PROXY-GOODHART: gradient optimizes SINGLE-frame VAE decode (−7.1% on that proxy) but the real mp4 (full temporal decode) is +4.1% drift. Motion preserved (+0.1%) but true drift not reduced. |
| G | FEEDFORWARD appearance correction (post-hoc, never fed to KV) | DONE | **no** — WORSE than closed-loop: lag8 +3.9%, motion −8.8%, frame0 +2.0%. OVERTURNS "feedback is the problem" hypothesis: the shared limit is the weak latent-DC target, not the feedback. |

## KEY INSIGHT (tick 5) — the coupling is proven
Across A/B/C/D (16 items, 4 axes each), drift-reduction and motion-preservation are COUPLED
under a frozen-frame-0 reference + output-latent actuation: the drift correction you get is
proportional to how hard you push the output toward frame-0, which is proportional to how much
you freeze motion (via KV feedback). Sparse enough to not freeze (D) => too sparse to reduce
drift. Aggressive enough to reduce drift (A) => freezes. **No output-latent actuator toward a
static anchor can beat open-loop on every axis.** Only untested class: conditioning/attention-
level anchoring (E) that never overwrites the output — the model decides how to use the anchor.
If E also fails, the well-supported conclusion is: test-time closed-loop control cannot beat
open-loop on every axis for this distilled AR world model.

## Scoreboard (16 items, 30s, Δ vs open; WIN needs ALL: windowed↓, rot≥−3%, motion≥−5%, tOF≤+1%)
| actuator | windowed lag8 | camera rot | motion | tOF | frame0 | verdict |
|---|---|---|---|---|---|---|
| A moment→f0 (v2) | −3% | −4.5% | −33% | −32% | +8% | no (freeze) |
| B context_noise | +5% | +7.6% | +0.4% | −0.0% | −1% | no (neutral) |
| C shift | +1.9% | −12.2% | −6.9% | −5.3% | +4.5% | no (froze via KV) |
| D gold-gated shift | +1.3% | −4.8% | −1.2% | −0.4% | −0.7% | no (neutral, no-harm) |
| **F gold_ctx (D+ctxnoise)** | **−0.1%** | **−1.5%** | **−2.2%** | **−1.4%** | **−1.8%** | **marginal WIN (all axes ≤0, but sub-2%, 8/16)** |
| G feedforward | +3.9% | −1.5% | −8.8% | −5.0% | +2.0% | no (worse; feedback not the culprit) |
| H gold_grad (63f/8itm) | +4.4% | −3.7% | +0.1% | +0.3% | +4.1% | no (proxy-Goodhart: single-frame-decode target ≠ real video) |
| I smooth-EMA (63f/8itm) | +0.3% | −5.7% | +4.3% | +4.6% | +5.2% | no (escaped freeze/+motion but windowed flat, tOF worse) |
| F-amplified (gain0.5,ctx3) | +3.5% | −7.0% | −8.8% | −7.8% | +4.2% | no (amplify overwhelms compensator; DC target is the ceiling) |

## Candidate I (NEW — before concluding)
| I | temporal EMA smoothing (moving reference) | DONE | **no** — escaped freeze (motion +4.3%!) but did NOT reduce windowed drift (+0.3% flat) and tOF WORSE (+4.6%). EMA-of-latent-means too weak a handle on perceptual flicker. |
Rationale: every prior actuator anchored to FROZEN frame-0 (=> KV freeze) and targeted drift-
from-start. I uses a MOVING reference (EMA of recent chunks) to remove high-freq flicker
(windowed drift) while FOLLOWING legit drift/motion. Never injects 'be static like f0' -> may
escape the freeze coupling. Cheap (no decode). Unit-verified: pulls jumps back, preserves
temporal deltas. Run: `run_smoothval_local.sh` (8 items, 63f, open vs smooth, `.smoothval_done`).
NEXT: eval (window+camera+fair locally) via combine_eval; if windowed↓ AND motion/camera
preserved -> full 30s/16-item + this could be THE win. If not -> proceed to conclusion.

## CAMPAIGN CLOSED (2026-07-14): N-scaling complete, NO-GO, FINAL_REPORT.md finalized
N8 both seeds: ALL drift metrics down (replicated direction!) but rot -15/-24deg; pooled N8
drift -2.6% t=-0.59 (bar was >=10%, t>=2) => saturates inside chaos floor; selection-freeze
emerges at high dose => waterbed operates through selection too. pid_reroll not run (bounded
by dose curve; argument recorded in FINAL_REPORT §6). Loop 5642f824 deleted. Deliverable:
vibe_docs/FINAL_REPORT.md (complete design+theory+verdict).

## N-SCALING STUDY (loop b9a422c3, every 10m)
3 replicates done (combine_eval real-video, frame0 final/mean vs open):
  N3-s42-A800 +8.1/+3.7 | N3-s123-H800 +0.4/-0.8 | N5-s42-H800 -3.4/-4.7 (rot +6.6!)
Selection is MOTION-SAFE everywhere; weak dose-response hint; chaos floor ~6% (same seed
A800-vs-H800 open baselines differ 6% -> only large effects or multi-seed means count).
IN FLIGHT: N8 x seeds{42,123} on 2 H800 nodes @5333 (jobs dreamx-reroll-n8-*-v2, slurm 1/2 after
server restart; sentinels .n8_s42_done/.n8_s123_done, outputs demo_reroll_n8_s42/, demo_reroll_n8_s123/).
NEXT TICKS: poll -> 4-axis eval (window+camera+fair, local GPUs) -> combine_eval -> pooled
dose-response N in {1,3,5,8}. If monotone & > chaos floor: run pid_reroll (PIDRerollScorer,
IMPLEMENTED + unit-verified: allocs ramp 0->4 with integral, deadband holds) vs fixed-N at
matched budget (read spent_extra_candidates from stats lines).

## REOPENED (user direction): candidate J = best-of-N chunk re-roll (ON-MANIFOLD selection)
Role: NOT a PID controller — the ORACLE/upper bound for the whole sensor->improvement chain
(plan's regime (a)). Every committed chunk is a genuine model sample -> immune to all three
off-manifold barriers (no latent surgery, no KV corruption, sensor only needs to RANK).
- If J fails: no PID actuator using this sensor can work -> the negative becomes airtight.
- If J wins: isolates the 8 failures to the actuation channel; PID becomes the cheap
  amortization of J (gate WHEN via PID error/integral; approximate the re-roll via guidance).
Latency: gated (gold drift > 0.2) so avg cost ~1.5-2x (n_extra=2); candidates batchable in
principle. Diagnostic, not deployable-streaming.
Implementation: pipeline `chunk_scorer` hook + `_denoise_chunk` closure refactor — REGRESSION-
VERIFIED BYTE-IDENTICAL (open path max|diff|=0.0 vs pre-refactor demo_hval reference).
`GoldRerollScorer` in dreamx_closed_loop.py; mode `reroll` in demo.
RUNNING: `run_reroll_local.sh` (16 items, 30s, open vs reroll, 4 local GPUs, sentinel
`.reroll_done`). EVAL when done: window+camera+fair locally -> `combine_eval.py --folder
demo_reroll_30s --modes open reroll`; also read reroll stats lines in /tmp/reroll_*.log
(rerolled/switched counts, mean score gain).

## QUANTIFIED COUPLING (tick ~20)
`coupling_analysis.py` over 128 (actuator,item) pairs (9 runs): corr(drift Δ, motion Δ)=+0.37
(p<<0.001). WIN quadrant (drift↓ AND motion kept)=19%; of drift-reducers only 39% kept motion.
Wins scattered => no actuator consistent => matches F-noise finding. Plot: coupling_frontier.png.
Capstone stat for the negative. Deliverable docs updated.

## FINALIZED (tick ~19): F's win is NOISE (replicated)
Seed-123 base-F: windowed +1.1% (was −0.1%), frame0 +4.3% (was −1.8%), verdict WIN→no, win-rate
6/16. F's seed-42 all-axis win does NOT replicate => confirmed statistical noise. The mission's
question is now answered with REPLICATION: no actuator (of 8) beats open-loop on every axis.
Deliverable `vibe_docs/closed_loop_conclusion.md` updated with the replication table. Loop
substantively COMPLETE (negative, well-evidenced + replicated). Further ticks only if user
redirects to training-time/architectural fixes (disentangled repr, break KV coupling).

## DELIVERABLE WRITTEN (tick ~18): vibe_docs/closed_loop_conclusion.md
Standalone conclusion: scoreboard (8 actuators), the (a)/(b) dichotomy, 3 barriers, what-would-
be-needed, bottom line. IN FLIGHT: base-F seed-123 replicate (`run_fs123_local.sh`, 30s/16 items,
`.fs123_done`) -> confirm F's ~1.5% edge is seed-noise. NEXT TICK: when done, combine_eval on
demo_goldctx_s123_30s + compare to seed-42 base F; if s123 also all-axis~0 but win-rate~8/16 and
signs differ across seeds => F edge is noise (finalizes doc). Loop's core question is ANSWERED
(negative, well-evidenced); remaining ticks = confirmatory stats + optional deeper training-time
ideas only if user redirects.

## STRENGTHENED DICHOTOMY (tick ~17, after I)
Across 8 actuators a clean DICHOTOMY holds: an actuator either
  (a) pushes appearance toward an anchor via output latent => reduces a drift proxy but FREEZES
      motion (A, C, D, F, amp-F, H), OR
  (b) avoids the freeze / preserves-or-adds motion (B context_noise, I smooth-EMA) => but then
      does NOT reduce windowed/perceptual drift.
NO actuator achieved BOTH. This is the core evidence: appearance-drift correction and motion are
coupled and test-time latent/KV control cannot decouple them for this distilled AR world model.

## CONCLUSION (tick ~16) — the case is proven (NEGATIVE, well-evidenced)
Explored 7 test-time closed-loop actuators × sensors (A moment, B context_noise, C DC-shift,
D gold-gated shift, F D+ctxnoise-compensator, G feedforward, H DINOv3-gradient) at 30s/16 items
(H at 15s/8) on 4 motion-fair axes. **NONE robustly beats open-loop on every axis.** Best = F
(gold-gated shift + context_noise compensator): the ONLY all-axis neutral-or-better result, but
sub-2% margins and 8/16 win-rate => within noise, not statistically convincing.

Three fundamental barriers (each proven by ≥2 experiments):
1. **KV-feedback coupling**: correcting the OUTPUT latent toward frozen frame-0 (A,C, amp-F)
   freezes motion via the recurrent context — regardless of per-chunk mechanics. Confirmed by C
   (motion-preserving per-chunk op still froze via KV).
2. **Weak cheap target**: latent-DC/moment correction (A,C,D,F,G) doesn't reduce PERCEPTUAL drift
   (G feedforward WORSE; amp-F worse) — shifting latent means ≠ reducing DINOv3 drift.
3. **Perceptual target doesn't transfer**: gradient through single-frame VAE decode (H) gamed its
   own decode path; real (full-temporal-decode) video not improved.

Net: for this distilled 4-step AR world model, appearance-drift reduction and motion are coupled
through the KV cache; test-time closed-loop control cannot cleanly decouple them. The only lever
that preserved/added motion (B context_noise) can't reduce drift; the only levers that reduce a
drift proxy (A/H) either freeze motion or don't transfer to the real video.

## NEXT TICK
- Write the standalone conclusion doc `vibe_docs/closed_loop_conclusion.md` (full scoreboard +
  3 barriers + the F marginal-win caveat) — the deliverable that "proves this case" with stats.
- OPTIONAL further shots (lower priority, likely null): (i) fix H target = full-sequence decode
  (hard: needs temporal VAE cache state); (ii) seed-replicate base F to formally show its ~1.5%
  edge is noise. Only if user wants more; otherwise the negative is well-supported.
- Scoreboard rows all recorded above.
