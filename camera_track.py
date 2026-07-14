"""
camera_track.py — camera-trajectory tracking: the one metric a frozen video cannot fake.

Commanded camera motion comes from the item's action trajectory (generate_trajectory_from_json).
Realized camera motion is estimated from the generated video by monocular visual odometry
(ORB features + essential matrix + recoverPose) between sampled frames.

Primary metric (freeze-proof, convention-robust):
  rot_realized_deg   = sum of per-step |relative rotation| recovered from the video (deg).
                       A frozen / static video -> ~0, regardless of appearance drift.
  rot_commanded_deg  = same sum from the commanded trajectory.
  track_ratio        = rot_realized / rot_commanded   (1 = tracks magnitude; ~0 = frozen).
  vo_fail_rate       = fraction of steps where VO found no reliable motion.

Intrinsics from configs/dreamx-ar: fx=0.505*W, fy=0.898*H, principal point at center.
Reads the mp4s written by demo_closed_loop.py. cv2 CPU; shardable.
"""

import argparse, glob, json, os
import numpy as np
import cv2
from torchvision.io import read_video

from utils.trajectory_processor import generate_trajectory_from_json, Camera


def geodesic_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))))


def commanded_rotations(item, num_pixel_frames, sample_idx):
    spec = list(zip([a.lower() for a in item["action_seq"]], item["action_speed_list"]))
    _, cam_np, _ = generate_trajectory_from_json(spec, num_frames=num_pixel_frames, return_cam_params=True)
    cams = [Camera(cam_np[i].tolist()) for i in range(cam_np.shape[0])]
    R = [np.array(c.w2c_mat)[:3, :3] for c in cams]           # world->cam rotation per frame
    tot = 0.0
    for a, b in zip(sample_idx[:-1], sample_idx[1:]):
        rel = R[b] @ R[a].T
        tot += geodesic_deg(rel)
    return tot


def realized_rotations(frames_gray, K):
    """Sum of |relative rotation| recovered by ORB+essential between consecutive frames."""
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    tot, fails = 0.0, 0
    for i in range(1, len(frames_gray)):
        g0, g1 = frames_gray[i - 1], frames_gray[i]
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
            fails += 1; continue
        matches = bf.knnMatch(d0, d1, k=2)
        good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
        if len(good) < 15:
            fails += 1; continue
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            fails += 1; continue
        _, R, _, _ = cv2.recoverPose(E, p0, p1, K)
        tot += geodesic_deg(R)
    return tot, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--data_path", default="configs/dreamx/eval.json")
    ap.add_argument("--modes", nargs="+", default=["open", "pid", "pid2"])
    ap.add_argument("--stride", type=int, default=8, help="sample every k-th pixel frame for VO")
    ap.add_argument("--resize_w", type=int, default=640)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    items = {os.path.splitext(os.path.basename(it["image_path"]))[0]: it
             for it in json.load(open(args.data_path))}

    names = sorted(os.path.basename(f).replace("_drift.json", "")
                   for f in glob.glob(os.path.join(args.folder, "*_drift.json")))
    names = names[args.shard_id::args.num_shards]

    rows = []
    for name in names:
        if name not in items:
            continue
        row = {"name": name}
        cmd_cache = None
        for m in args.modes:
            mp4 = os.path.join(args.folder, f"{name}_{m}.mp4")
            if not os.path.exists(mp4):
                continue
            v, _, _ = read_video(mp4, pts_unit="sec", output_format="THWC")   # uint8 [T,H,W,C] RGB
            T, H, W = v.shape[0], v.shape[1], v.shape[2]
            sample_idx = list(range(0, T, args.stride))
            sw = args.resize_w; sh = int(H * sw / W)
            gray = [cv2.cvtColor(cv2.resize(v[t].numpy(), (sw, sh)), cv2.COLOR_RGB2GRAY) for t in sample_idx]
            fx, fy = 0.505 * sw, 0.898 * sh
            K = np.array([[fx, 0, sw / 2.0], [0, fy, sh / 2.0], [0, 0, 1.0]], dtype=np.float64)
            r_real, fails = realized_rotations(gray, K)
            if cmd_cache is None:
                cmd_cache = commanded_rotations(items[name], T, sample_idx)
            ratio = r_real / cmd_cache if cmd_cache > 1e-6 else 0.0
            row[m] = {"rot_realized_deg": r_real, "rot_commanded_deg": cmd_cache,
                      "track_ratio": ratio, "vo_fail_rate": fails / max(1, len(gray) - 1)}
        rows.append(row)
        print(f"[{name[:24]:24}] " + "  ".join(
            f"{m}: real={row[m]['rot_realized_deg']:.1f}deg ratio={row[m]['track_ratio']:.2f}"
            for m in args.modes if m in row), flush=True)

    with open(os.path.join(args.folder, f"camera_track_shard{args.shard_id}.json"), "w") as f:
        json.dump({"per_item": rows}, f, indent=2)
    print(f"wrote camera_track_shard{args.shard_id}.json")


if __name__ == "__main__":
    main()
