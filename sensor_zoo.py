"""
sensor_zoo.py — candidate reflection signals for DreamX-World closed-loop control.

Two families:
  CHEAP  (latent-space, NO decode) — candidate in-loop sensors / actuation targets.
  GOLD   (decoded pixels, perceptual/identity space) — trusted evaluation & referee.

The whole point of validate_sensors.py is to decide which CHEAP sensor (if any) is a
faithful proxy for a GOLD sensor, BEFORE we ever close the loop. A cheap sensor that
lives in the SAME space the actuator manipulates (latent moments) can be gamed to
zero error while the real appearance drifts — Goodhart. The GOLD sensors live in a
space the latent actuator cannot directly touch, so they can't be gamed.

Every sensor is a callable that returns a 1-D torch vector (an embedding / feature).
Error between two readings is `sensor_error(a, b)` = 1 - cosine (bounded, [0,2]),
except scalar-distance sensors (DreamSim) which return their native distance.

Extractors load lazily; a missing dependency disables that sensor with a warning
instead of crashing the whole harness.
"""

import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F

# Local DINOv3 checkpoint (HF transformers format). Override with DINOV3_PATH.
_DINOV3_DEFAULT = "/home/ma-user/work/dataset/xiaoyi_video_env/tmp/dinov3-vitb16-pretrain-lvd1689m"


# ─────────────────────────── error metric ───────────────────────────

def sensor_error(a, b):
    """1 - cosine similarity between two feature vectors. 0 == identical direction."""
    a = a.float().flatten()
    b = b.float().flatten()
    return float(1.0 - F.cosine_similarity(a, b, dim=0, eps=1e-8))


# ─────────────────────────── CHEAP sensors (latent) ───────────────────────────
# Input: latent chunk/frame [B, T, C, H, W] (C=48 for DreamX). Output: 1-D vector.

def cheap_moments(latent):
    """Per-channel mean+std over the chunk. 2C dims. == postprocess.py's Lab stats,
    but in latent space. This is ALSO actuator (A)'s target -> collocation risk."""
    x = latent.float()
    mean = x.mean(dim=(0, 1, 3, 4))
    std = x.std(dim=(0, 1, 3, 4))
    return torch.cat([mean, std])


def cheap_mean(latent):
    """Per-channel mean only. C dims. Weaker; a baseline for the ablation."""
    return latent.float().mean(dim=(0, 1, 3, 4))


def cheap_pooled(latent, size=4):
    """Spatially pooled latent, flattened. Keeps coarse structure the moments throw
    away — a candidate that is NOT fully collocated with the moment actuator."""
    x = latent.float().mean(dim=1)                      # [B,C,H,W] avg over frames
    x = F.adaptive_avg_pool2d(x, size)                  # [B,C,size,size]
    return x.flatten()


CHEAP_SENSORS = {
    "latent_moments": cheap_moments,
    "latent_mean": cheap_mean,
    "latent_pooled4": cheap_pooled,
}


# ─────────────────────────── GOLD sensors (decoded pixels) ───────────────────────────
# Input: an RGB frame tensor [3, H, W] in [0,1]. Output: 1-D embedding (or scalar dist).

class _DINOv2:
    """DINOv2 ViT-B/14 CLS token. Perceptual scene-appearance identity; robust,
    lives in a space the latent actuator cannot directly manipulate."""
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    @torch.no_grad()
    def __call__(self, frame):
        x = frame.to(self.device)
        x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
        x = (x[0] - self.mean) / self.std
        return self.model(x.unsqueeze(0)).squeeze(0).cpu()


class _ArcFace:
    """InsightFace embedding — the *identity* sensor. Naturally invariant to lighting
    /pose (criterion 5), so a better setpoint than raw appearance when a face is present.
    Returns None when no face is detected (sensor dropout — handled upstream)."""
    def __init__(self, device):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(512, 512))

    @torch.no_grad()
    def __call__(self, frame):
        img = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
        faces = self.app.get(np.ascontiguousarray(img))
        if not faces:
            return None
        return torch.from_numpy(faces[0].normed_embedding).float()


class _DreamSim:
    """DreamSim perceptual distance to a stored reference. Returns a SCALAR distance
    (not an embedding) — tracks human perceptual similarity well."""
    def __init__(self, device):
        from dreamsim import dreamsim
        self.model, self.preprocess = dreamsim(pretrained=True, device=device)
        self.device = device
        self._ref = None

    def set_reference(self, frame):
        self._ref = self._prep(frame)

    def _prep(self, frame):
        from PIL import Image
        img = Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        return self.preprocess(img).to(self.device)

    @torch.no_grad()
    def __call__(self, frame):
        # returns a 1-vector holding the scalar distance to the set reference
        d = self.model(self._ref, self._prep(frame))
        return torch.tensor([float(d)])


class _DINOv3:
    """DINOv3 ViT-B/16 CLS token (local HF transformers checkpoint). Strictly stronger
    perceptual/scene-appearance referee than DINOv2; same ImageNet norm, 224x224.
    Lives in a space the latent moment-actuator cannot directly manipulate -> un-gameable."""
    def __init__(self, device):
        from transformers import AutoModel
        path = os.environ.get("DINOV3_PATH", _DINOV3_DEFAULT)
        self.device = device
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    @torch.no_grad()
    def __call__(self, frame):
        x = frame.to(self.device).float()
        x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
        x = (x[0] - self.mean) / self.std
        out = self.model(pixel_values=x.unsqueeze(0))
        return out.last_hidden_state[0, 0].float().cpu()   # CLS token

    def embed_grad(self, frame):
        """Grad-enabled CLS embedding (stays on device, no no_grad) for gradient actuation."""
        x = frame.to(self.device).float()
        x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
        x = (x[0] - self.mean) / self.std
        out = self.model(pixel_values=x.unsqueeze(0))
        return out.last_hidden_state[0, 0].float()


_GOLD_BUILDERS = {"dino": _DINOv2, "dino3": _DINOv3, "arcface": _ArcFace, "dreamsim": _DreamSim}


def load_gold_sensors(names, device):
    """Instantiate requested gold sensors; skip (with warning) any whose deps are missing."""
    gold = {}
    for n in names:
        try:
            gold[n] = _GOLD_BUILDERS[n](device)
        except Exception as e:  # noqa: BLE001 — deliberately tolerant
            warnings.warn(f"[sensor_zoo] gold sensor '{n}' unavailable, skipping: {e}")
    return gold
