import os
import glob
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_intrinsics(camera_info_yaml_path: str) -> np.ndarray:
    with open(camera_info_yaml_path, "r") as f:
        docs = list(yaml.safe_load_all(f))
    if not docs:
        raise RuntimeError(f"No YAML docs found in {camera_info_yaml_path}")
    data = docs[0]
    K = np.array(data["k"], dtype=np.float64).reshape(3, 3)
    return K


def load_images(images_dir: str):
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    if not paths:
        raise RuntimeError(f"No .png images found in: {images_dir}")
    return paths


def main():
    bag_root = "/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/minicity_3lap_visual/minicity_3lap_visual"
    images_dir = os.path.join(bag_root, "extracted/images")
    cam_yaml = os.path.join(bag_root, "extracted/meta/camera_info.yaml")
    out_dir = "/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/monocular_vo/output"
    os.makedirs(out_dir, exist_ok=True)

    K = load_intrinsics(cam_yaml)
    img_paths = load_images(images_dir)
    print(f"[INFO] Loaded {len(img_paths)} images")

    # ---- ORB tuning (more features, better distribution) ----
    orb = cv2.ORB_create(
        nfeatures=4000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=19,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=10,
    )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Global pose (scale unknown)
    T = np.eye(4, dtype=np.float64)
    traj = [T[:3, 3].copy()]

    # Logs
    frame_ids = []
    good_matches_log = []
    inliers_log = []
    skipped = 0

    # Visualization setup (saves a few frames)
    viz_dir = os.path.join(out_dir, "matches_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_viz_every = 150  # save 1 image every N frames

    prev = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    if prev is None:
        raise RuntimeError(f"Failed to read first image: {img_paths[0]}")
    kp1, des1 = orb.detectAndCompute(prev, None)

    for i in range(1, len(img_paths)):
        img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue

        kp2, des2 = orb.detectAndCompute(img, None)
        if des1 is None or des2 is None:
            skipped += 1
            kp1, des1 = kp2, des2
            prev = img
            continue

        matches_knn = bf.knnMatch(des1, des2, k=2)

        # Lowe ratio test
        good = []
        for pair in matches_knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 80:
            skipped += 1
            frame_ids.append(i)
            good_matches_log.append(len(good))
            inliers_log.append(0)
            kp1, des1 = kp2, des2
            prev = img
            continue

        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        if E is None:
            skipped += 1
            frame_ids.append(i)
            good_matches_log.append(len(good))
            inliers_log.append(0)
            kp1, des1 = kp2, des2
            prev = img
            continue

        # Count inliers from findEssentialMat mask
        inliers = int(mask.sum()) if mask is not None else 0

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t[:, 0]  # unit direction

        T = T @ np.linalg.inv(T_rel)
        traj.append(T[:3, 3].copy())

        frame_ids.append(i)
        good_matches_log.append(len(good))
        inliers_log.append(inliers)

        # Save match visualization occasionally
        if i % save_viz_every == 0:
            img_color = cv2.imread(img_paths[i])
            prev_color = cv2.imread(img_paths[i-1])
            if img_color is not None and prev_color is not None:
                vis = cv2.drawMatches(
                    prev_color, kp1, img_color, kp2,
                    good[:100], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imwrite(os.path.join(viz_dir, f"matches_{i:06d}.png"), vis)

        if i % 100 == 0:
            print(f"[INFO] Processed {i}/{len(img_paths)-1} | good={len(good)} | inliers={inliers}")

        kp1, des1 = kp2, des2
        prev = img

    traj = np.array(traj)
    np.save(os.path.join(out_dir, "trajectory.npy"), traj)

    # Save stats as CSV
    stats_path = os.path.join(out_dir, "vo_stats.csv")
    with open(stats_path, "w") as f:
        f.write("frame,good_matches,inliers\n")
        for fr, gm, il in zip(frame_ids, good_matches_log, inliers_log):
            f.write(f"{fr},{gm},{il}\n")

    # Plot trajectory
    plt.figure()
    plt.plot(traj[:, 0], traj[:, 2])
    plt.xlabel("X (arb. units)")
    plt.ylabel("Z (arb. units)")
    plt.title("Monocular VO Trajectory (Scale Unknown)")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "trajectory.png"), dpi=200)

    # Plot matches vs time
    plt.figure()
    plt.plot(frame_ids, good_matches_log, label="good matches")
    plt.plot(frame_ids, inliers_log, label="inliers (E-RANSAC)")
    plt.xlabel("Frame index")
    plt.ylabel("Count")
    plt.title("VO Tracking Quality Over Time")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "matches_vs_time.png"), dpi=200)

    print(f"[DONE] Saved trajectory + stats to: {out_dir}")
    print(f"[DONE] Skipped frames: {skipped}")
    print(f"[DONE] Stats CSV: {stats_path}")
    print(f"[DONE] Viz images: {viz_dir}")

    plt.show()


if __name__ == "__main__":
    main()
