"""
dreamx_closed_loop.py — test-time closed-loop controller for DreamX-World AR rollout.

Wiring: pipeline.inference(..., chunk_callback=ctrl). The callback fires AFTER a chunk's
denoised_pred is produced and BEFORE the context-noise rerun writes it into the KV cache,
so the correction propagates to every future chunk.

Sensor (in-loop, cheap, no decode): per-channel latent moments [mean_C ; std_C]  (== the
validated `latent_moments` proxy). Setpoint = moments of frame-0 (initial_latent).

Actuator (A): re-set the chunk latent's per-channel moments toward a PID target via a
per-channel affine map.

v2 additions to avoid punishing *reasonable* drift (legitimate scene evolution / motion):
  * deadband     — leave chunks whose drift < deadband uncorrected (only act on excess drift).
  * I-dominant   — smaller kp, larger ki: fight the slow *accumulated* drift ramp, not every
                   instantaneous per-chunk change (the P term is what fights legitimate motion).
Gating also freezes the integrator inside the deadband (anti-windup).

The controller NEVER sees DINOv3 — the gold metric is used only for evaluation.
"""

import torch


def latent_moments(latent):
    """[B,T,C,H,W] -> ([mean_C], [std_C]) per channel over B,T,H,W."""
    x = latent.float()
    mean = x.mean(dim=(0, 1, 3, 4))
    std = x.std(dim=(0, 1, 3, 4))
    return mean, std


class GoldRerollScorer:
    """Best-of-N chunk re-roll (ON-MANIFOLD closed loop) — `chunk_scorer` for
    CausalCameraInferencePipeline.inference.

    Per chunk: decode the last latent frame, measure gold drift (1 - cos DINOv3 vs frame-0
    anchor). If drift > gold_thresh, regenerate the chunk up to n_extra times with fresh noise
    (via the reroll_fn the pipeline provides) and COMMIT THE BEST-SCORING GENUINE SAMPLE.
    Nothing externally modified ever enters the KV cache — selection, not actuation. This is
    immune to the off-manifold barriers: no latent surgery, no KV corruption, and the sensor
    only needs to RANK candidates (DINOv3's validated strength).
    """
    def __init__(self, initial_latent, decode_fn, dino, gold_thresh=0.2, n_extra=2):
        self.decode_fn, self.dino = decode_fn, dino
        self.gold_thresh, self.n_extra = gold_thresh, n_extra
        self.anchor_emb = dino(decode_fn(initial_latent[:, :1]))
        self.drift_log = []      # (chunk_index, first-sample drift)
        self.choice_log = []     # (chunk_index, chosen_k, best_score, first_score)
        self.gate_log = []       # 1.0 = re-rolled this chunk, 0.0 = kept first sample

    def _score(self, pred):
        emb = self.dino(self.decode_fn(pred[:, -1:]))
        a = self.anchor_emb.float().flatten(); b = emb.float().flatten()
        return float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8))

    def __call__(self, first_pred, reroll_fn, chunk_index=0):
        s0 = self._score(first_pred)
        self.drift_log.append((chunk_index, s0))
        if s0 <= self.gold_thresh:
            self.choice_log.append((chunk_index, 0, s0, s0))
            self.gate_log.append(0.0)
            return first_pred
        best, best_s, best_k = first_pred, s0, 0
        for k in range(1, self.n_extra + 1):
            cand = reroll_fn()
            s = self._score(cand)
            if s < best_s:
                best, best_s, best_k = cand, s, k
        self.choice_log.append((chunk_index, best_k, best_s, s0))
        self.gate_log.append(1.0)
        return best

    def stats(self):
        rerolled = [c for c in self.choice_log if c[3] > self.gold_thresh]
        switched = [c for c in rerolled if c[1] > 0]
        gain = [c[3] - c[2] for c in rerolled]
        import numpy as np
        return {"chunks": len(self.choice_log), "rerolled": len(rerolled),
                "switched": len(switched),
                "mean_score_gain_when_rerolled": float(np.mean(gain)) if gain else 0.0}


class PIDRerollScorer(GoldRerollScorer):
    """PI-SCHEDULED test-time compute: the PID law allocates the candidate budget per chunk
    instead of a fixed gate+N. The control law never touches latents — it decides how much
    correction compute each chunk deserves, based on instantaneous AND accumulated excess drift:

        e_k      = gold drift of chunk k's first sample
        excess_k = max(0, e_k - deadband)                 # legit change tolerated
        I_k      = leak * I_{k-1} + excess_k              # leaky integral (anti-windup)
        n_k      = clamp(round(kp*excess_k + ki*I_k), 0, n_max)   # candidates for chunk k

    Feedback closure: a successful re-roll lowers subsequent e_k, draining the integral —
    the controller backs off automatically after correcting. Compare vs fixed-N at MATCHED
    average compute (log `spent` = total extra candidates)."""
    def __init__(self, initial_latent, decode_fn, dino,
                 kp=4.0, ki=2.0, leak=0.9, deadband=0.2, n_max=7):
        super().__init__(initial_latent, decode_fn, dino, gold_thresh=deadband, n_extra=0)
        self.kp, self.ki, self.leak, self.deadband, self.n_max = kp, ki, leak, deadband, n_max
        self.I = 0.0
        self.n_log = []          # (chunk_index, e, I, n_allocated)
        self.spent = 0           # total extra candidates generated

    def __call__(self, first_pred, reroll_fn, chunk_index=0):
        e = self._score(first_pred)
        self.drift_log.append((chunk_index, e))
        excess = max(0.0, e - self.deadband)
        self.I = self.leak * self.I + excess
        n = int(min(max(round(self.kp * excess + self.ki * self.I), 0), self.n_max))
        self.n_log.append((chunk_index, e, self.I, n))
        if n == 0:
            self.choice_log.append((chunk_index, 0, e, e))
            self.gate_log.append(0.0)
            return first_pred
        best, best_s, best_k = first_pred, e, 0
        for k in range(1, n + 1):
            cand = reroll_fn()
            s = self._score(cand)
            self.spent += 1
            if s < best_s:
                best, best_s, best_k = cand, s, k
        self.choice_log.append((chunk_index, best_k, best_s, e))
        self.gate_log.append(1.0)
        return best

    def stats(self):
        base = super().stats()
        import numpy as np
        ns = [x[3] for x in self.n_log]
        base.update({"spent_extra_candidates": self.spent,
                     "mean_n": float(np.mean(ns)) if ns else 0.0,
                     "max_n": int(max(ns)) if ns else 0})
        return base


class LeakyPID:
    """Per-element leaky-integral PID on a vector error. leak<1 = forgetting factor
    (anti-windup for unbounded horizons); I is clamped as a second guard. `gate`=0 skips
    both actuation and integral accumulation for that step (deadband anti-windup)."""
    def __init__(self, kp=0.2, ki=0.1, kd=0.0, leak=0.85, i_clamp=5.0):
        self.kp, self.ki, self.kd, self.leak, self.i_clamp = kp, ki, kd, leak, i_clamp
        self.I = None
        self.prev_y = None

    def step(self, e, y, gate=1.0):
        if self.I is None:
            self.I = torch.zeros_like(e)
            self.prev_y = y
        self.I = self.leak * self.I + gate * e          # gate=0 -> integral only leaks
        self.I = torch.clamp(self.I, -self.i_clamp, self.i_clamp)
        d = -(y - self.prev_y)                          # derivative on measurement
        self.prev_y = y
        u = self.kp * e + self.ki * self.I + self.kd * d
        return u * gate                                 # gate=0 -> no actuation


class DreamXClosedLoop:
    """chunk_callback for CausalCameraInferencePipeline.

    mode: 'pid' (P+I), 'p' (proportional only), 'i' (integral only) — for the moment actuator.
    actuator:
      'moment'   — actuator A: nudge the chunk's per-channel latent moments toward frame-0.
      'ctxnoise' — actuator B: raise pipeline.args.context_noise when drift is high, so the
                   line-226 KV rerun re-encodes the chunk at a higher noise timestep → the
                   model trusts its own drifted recent context less and leans on the sink
                   anchor + prompt. Non-destructive; returns denoised_pred UNCHANGED.
      'both'     — B plus a gentle moment nudge.
    gain_max: cap on the per-channel fraction moved toward the setpoint per chunk (safety).
    deadband: skip correction while cheap moment-drift (1-cos vs frame 0) < deadband.
    args_ref: the pipeline.args object (for actuator B to mutate context_noise per chunk).
    ctx_base/ctx_k/ctx_max/ctx_deadband: context_noise = clamp(base + k*max(0,drift-db), base, max).
    """
    def __init__(self, initial_latent, mode="pid", kp=0.2, ki=0.1, leak=0.85,
                 gain_max=0.3, min_std_ratio=0.25, deadband=0.0,
                 actuator="moment", args_ref=None,
                 ctx_base=0.1, ctx_k=1.0, ctx_max=0.4, ctx_deadband=0.05,
                 decode_fn=None, dino=None, decode_every=3, gold_thresh=0.15,
                 decode_grad_fn=None, grad_step=0.05,
                 smooth_beta=0.7, smooth_gain=0.3):
        r_mean, r_std = latent_moments(initial_latent)
        self.r_mean, self.r_std = r_mean, r_std          # setpoint (frame-0 moments)
        if mode == "p":
            ki = 0.0
        elif mode == "i":
            kp = 0.0
        self.pid_mean = LeakyPID(kp=kp, ki=ki, leak=leak)
        self.pid_std = LeakyPID(kp=kp, ki=ki, leak=leak)
        self.gain_max = gain_max
        self.min_std_ratio = min_std_ratio
        self.deadband = deadband
        self.actuator = actuator
        self.args_ref = args_ref
        self.ctx_base, self.ctx_k, self.ctx_max, self.ctx_deadband = ctx_base, ctx_k, ctx_max, ctx_deadband
        # gold_shift: decoded DINOv3 gate on a motion-preserving DC-shift
        self.decode_fn, self.dino = decode_fn, dino
        self.decode_every, self.gold_thresh = decode_every, gold_thresh
        self.decode_grad_fn, self.grad_step = decode_grad_fn, grad_step
        self.anchor_emb = None
        self.anchor_emb_dev = None                       # on-device anchor for gradient actuator
        self.gold_drift = 0.0
        if actuator in ("gold_shift", "gold_ctx") and decode_fn is not None and dino is not None:
            self.anchor_emb = dino(decode_fn(initial_latent[:, :1]))
        if actuator == "gold_grad" and decode_grad_fn is not None and dino is not None:
            self.anchor_emb_dev = dino.embed_grad(decode_grad_fn(initial_latent[:, :1])).detach()
        self.drift_log = []                              # cheap moment-drift per chunk
        self.gate_log = []                               # 1 = corrected, 0 = deadband (free)
        self.ctx_log = []                                # context_noise applied per chunk
        self.gold_log = []                               # decoded gold drift when sampled
        # temporal EMA smoothing (moving reference, NOT frozen frame-0)
        self.smooth_beta, self.smooth_gain = smooth_beta, smooth_gain
        self.ema_mean = None

    def _apply(self, latent, tgt_mean, tgt_std):
        x = latent.float()
        C = x.shape[2]
        cur_mean = x.mean(dim=(0, 1, 3, 4), keepdim=True)
        cur_std = x.std(dim=(0, 1, 3, 4), keepdim=True) + 1e-6
        tm = tgt_mean.view(1, 1, C, 1, 1)
        ts = torch.clamp(tgt_std, min=self.min_std_ratio * 1e-6).view(1, 1, C, 1, 1)
        out = (x - cur_mean) * (ts / cur_std) + tm
        return out.to(latent.dtype)

    def __call__(self, denoised_pred, chunk_index=0, current_start_frame=0):
        y_mean, y_std = latent_moments(denoised_pred)
        e_mean = self.r_mean - y_mean
        e_std = self.r_std - y_std

        yv = torch.cat([y_mean, y_std]); rv = torch.cat([self.r_mean, self.r_std])
        drift = float(1.0 - torch.nn.functional.cosine_similarity(yv, rv, dim=0, eps=1e-8))
        self.drift_log.append(drift)

        # --- Actuator I: temporal EMA smoothing (moving reference, NOT frozen frame-0) ---
        # Pull each chunk's per-channel mean gently toward an EMA of RECENT chunks (the moving
        # trend), removing high-frequency chunk-to-chunk appearance JUMPS (flicker / windowed
        # drift) while FOLLOWING legitimate slow drift + motion. DC-shift only (motion-preserving).
        # Never injects "be static like frame-0" -> should escape the KV freeze coupling.
        if self.actuator == "smooth":
            C = denoised_pred.shape[2]
            if self.ema_mean is None:
                self.ema_mean = y_mean.clone()
                self.gate_log.append(0.0)
                return denoised_pred
            u = self.smooth_gain * (self.ema_mean - y_mean)      # pull toward recent trend
            corrected_mean = y_mean + u
            self.ema_mean = self.smooth_beta * self.ema_mean + (1 - self.smooth_beta) * corrected_mean
            self.gate_log.append(1.0)
            return (denoised_pred.float() + u.view(1, 1, C, 1, 1)).to(denoised_pred.dtype)

        # --- Actuator B: adaptive context_noise (non-destructive) ---
        if self.actuator in ("ctxnoise", "both") and self.args_ref is not None:
            ctx = self.ctx_base + self.ctx_k * max(0.0, drift - self.ctx_deadband)
            ctx = float(min(max(ctx, self.ctx_base), self.ctx_max))
            self.args_ref.context_noise = ctx
            self.ctx_log.append(ctx)

        # Pure ctxnoise: no latent change.
        if self.actuator == "ctxnoise":
            self.gate_log.append(0.0)
            return denoised_pred

        # --- Actuator C: appearance-only DC-drift shift (motion-preserving) ---
        # The std-rescale in actuator A flattens temporal/spatial variance -> freezes motion.
        # C applies ONLY a per-channel additive shift (PID on the mean error), which preserves
        # every per-frame and per-pixel deviation (structure/motion) by construction. It also
        # only corrects the LOW-FREQUENCY appearance (DC) drift, leaving content free.
        if self.actuator == "shift":
            gate = 1.0 if drift >= self.deadband else 0.0
            self.gate_log.append(gate)
            u_mean = self.pid_mean.step(e_mean, y_mean, gate=gate)
            if gate == 0.0:
                return denoised_pred
            u_mean = torch.clamp(u_mean, -self.gain_max * e_mean.abs(), self.gain_max * e_mean.abs())
            C = denoised_pred.shape[2]
            return (denoised_pred.float() + u_mean.view(1, 1, C, 1, 1)).to(denoised_pred.dtype)

        # --- Actuator D: gold-GATED sparse motion-preserving shift ---
        # Only correct chunks where a decoded DINOv3 embedding confirms REAL appearance drift.
        # Sparse + verified -> far fewer corrections -> less cumulative motion suppression than
        # the ungated shift (C), while still catching genuine drift. Uses the DC-shift (no std
        # rescale) so the per-chunk op preserves motion.
        if self.actuator == "gold_shift":
            is_tick = (chunk_index % self.decode_every == 0)
            if is_tick and self.decode_fn is not None:
                emb = self.dino(self.decode_fn(denoised_pred[:, -1:]))
                a = self.anchor_emb.float().flatten(); b = emb.float().flatten()
                self.gold_drift = float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8))
                self.gold_log.append((chunk_index, self.gold_drift))
            # correct ONLY on decode ticks with confirmed drift -> sparse KV commits -> less
            # cumulative motion suppression than the continuous shift (C).
            gate = 1.0 if (is_tick and self.gold_drift > self.gold_thresh) else 0.0
            self.gate_log.append(gate)
            u_mean = self.pid_mean.step(e_mean, y_mean, gate=gate)
            if gate == 0.0:
                return denoised_pred
            u_mean = torch.clamp(u_mean, -self.gain_max * e_mean.abs(), self.gain_max * e_mean.abs())
            C = denoised_pred.shape[2]
            return (denoised_pred.float() + u_mean.view(1, 1, C, 1, 1)).to(denoised_pred.dtype)

        # --- Actuator F: gold-gated shift + context_noise MOTION COMPENSATOR ---
        # D reduces drift a little but costs camera motion (KV freeze). B (context_noise) ADDS
        # motion. F does both: gold-gated DC-shift to reduce drift, AND raises context_noise with
        # gold drift to inject motion back -> attempt drift-reduction AT preserved motion (attack
        # the coupling with an independent motion knob).
        if self.actuator == "gold_ctx":
            is_tick = (chunk_index % self.decode_every == 0)
            if is_tick and self.decode_fn is not None:
                emb = self.dino(self.decode_fn(denoised_pred[:, -1:]))
                a = self.anchor_emb.float().flatten(); b = emb.float().flatten()
                self.gold_drift = float(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-8))
                self.gold_log.append((chunk_index, self.gold_drift))
            if self.args_ref is not None:                       # motion compensator (every chunk)
                ctx = self.ctx_base + self.ctx_k * max(0.0, self.gold_drift - self.ctx_deadband)
                self.args_ref.context_noise = float(min(max(ctx, self.ctx_base), self.ctx_max))
                self.ctx_log.append(self.args_ref.context_noise)
            gate = 1.0 if (is_tick and self.gold_drift > self.gold_thresh) else 0.0
            self.gate_log.append(gate)
            u_mean = self.pid_mean.step(e_mean, y_mean, gate=gate)
            if gate == 0.0:
                return denoised_pred
            u_mean = torch.clamp(u_mean, -self.gain_max * e_mean.abs(), self.gain_max * e_mean.abs())
            C = denoised_pred.shape[2]
            return (denoised_pred.float() + u_mean.view(1, 1, C, 1, 1)).to(denoised_pred.dtype)

        # --- Actuator H: DINOv3-GRADIENT actuation (better target than latent-DC) ---
        # On decode ticks, backprop the DINOv3-distance-to-anchor through VAE+DINOv3 to the chunk's
        # last latent frame, and step along -grad: the minimal-norm latent change that actually
        # reduces PERCEPTUAL drift (vs blunt latent-DC). + ctxnoise motion compensator. Defensive:
        # any grad failure (OOM / no-grad VAE) -> no correction (never crashes a run).
        if self.actuator == "gold_grad":
            is_tick = (chunk_index % self.decode_every == 0)
            if self.args_ref is not None:                       # motion compensator every chunk
                ctx = self.ctx_base + self.ctx_k * max(0.0, self.gold_drift - self.ctx_deadband)
                self.args_ref.context_noise = float(min(max(ctx, self.ctx_base), self.ctx_max))
                self.ctx_log.append(self.args_ref.context_noise)
            if not is_tick or self.decode_grad_fn is None:
                self.gate_log.append(0.0)
                return denoised_pred
            try:
                with torch.enable_grad():
                    lat = denoised_pred[:, -1:].detach().float().requires_grad_(True)
                    emb = self.dino.embed_grad(self.decode_grad_fn(lat)).flatten()
                    a = self.anchor_emb_dev.float().flatten()
                    loss = 1.0 - torch.nn.functional.cosine_similarity(emb, a, dim=0, eps=1e-8)
                    self.gold_drift = float(loss.detach())
                    grad = torch.autograd.grad(loss, lat)[0]
                self.gold_log.append((chunk_index, self.gold_drift))
                if self.gold_drift <= self.gold_thresh:
                    self.gate_log.append(0.0)
                    return denoised_pred
                g = grad / (grad.norm() + 1e-8)
                step = self.grad_step * denoised_pred[:, -1:].float().norm()
                out = denoised_pred.float().clone()
                out[:, -1:] = out[:, -1:] - step * g
                self.gate_log.append(1.0)
                return out.to(denoised_pred.dtype)
            except Exception as e:                              # OOM / non-differentiable VAE
                self._grad_err = str(e)[:120]
                self.gate_log.append(0.0)
                return denoised_pred

        # --- Actuator A: moment nudge toward frame-0 ---
        # deadband: leave "reasonable" (small) drift uncorrected so legitimate scene
        # evolution / motion is not fought; only act once drift exceeds the band.
        gate = 1.0 if drift >= self.deadband else 0.0
        self.gate_log.append(gate)

        u_mean = self.pid_mean.step(e_mean, y_mean, gate=gate)
        u_std = self.pid_std.step(e_std, y_std, gate=gate)
        if gate == 0.0:
            return denoised_pred

        # safety: cap the per-channel move to at most gain_max of the way to setpoint
        u_mean = torch.clamp(u_mean, -self.gain_max * e_mean.abs(), self.gain_max * e_mean.abs())
        u_std = torch.clamp(u_std, -self.gain_max * e_std.abs(), self.gain_max * e_std.abs())

        return self._apply(denoised_pred, y_mean + u_mean, y_std + u_std)
