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
    data = docs[0]  # take first message
    k = data["k"]   # [fx,0,cx, 0,fy,cy, 0,0,1]
    K = np.array(k, dtype=np.float64).reshape(3, 3)
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
    print(f"[INFO] K=\n{K}")

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    T = np.eye(4, dtype=np.float64)
    traj = [T[:3, 3].copy()]

    prev = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    if prev is None:
        raise RuntimeError(f"Failed to read first image: {img_paths[0]}")
    kp1, des1 = orb.detectAndCompute(prev, None)

    for i in range(1, len(img_paths)):
        img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read {img_paths[i]}, skipping")
            continue

        kp2, des2 = orb.detectAndCompute(img, None)
        if des1 is None or des2 is None:
            kp1, des1 = kp2, des2
            prev = img
            continue

        matches_knn = bf.knnMatch(des1, des2, k=2)

        good = []
        for pair in matches_knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 80:
            print(f"[WARN] Frame {i}: too few matches ({len(good)}), skipping")
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
            print(f"[WARN] Frame {i}: findEssentialMat failed, skipping")
            kp1, des1 = kp2, des2
            prev = img
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # Relative transform (direction only; scale unknown)
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t[:, 0]

        # Accumulate pose: world_T_cam
        T = T @ np.linalg.inv(T_rel)
        traj.append(T[:3, 3].copy())

        if i % 100 == 0:
            print(f"[INFO] Processed {i}/{len(img_paths)-1}")

        kp1, des1 = kp2, des2
        prev = img

    traj = np.array(traj)
    np.save(os.path.join(out_dir, "trajectory.npy"), traj)

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 2])
    plt.xlabel("X (arb. units)")
    plt.ylabel("Z (arb. units)")
    plt.title("Monocular VO Trajectory (Scale Unknown)")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "trajectory.png"), dpi=200)
    plt.show()

    print(f"[DONE] Saved: {os.path.join(out_dir, 'trajectory.png')}")


if __name__ == "__main__":
    main()
import os
import glob
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_intrinsics(camera_info_yaml_path: str) -> np.ndarray:
    with open(camera_info_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    k = data["k"]
    K = np.array(k, dtype=np.float64).reshape(3, 3)
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
    print(f"[INFO] K=\n{K}")

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    T = np.eye(4, dtype=np.float64)
    traj = [T[:3, 3].copy()]

    prev = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    if prev is None:
        raise RuntimeError(f"Failed to read first image: {img_paths[0]}")
    kp1, des1 = orb.detectAndCompute(prev, None)

    for i in range(1, len(img_paths)):
        img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read {img_paths[i]}, skipping")
            continue

        kp2, des2 = orb.detectAndCompute(img, None)
        if des1 is None or des2 is None:
            kp1, des1 = kp2, des2
            prev = img
            continue

        matches_knn = bf.knnMatch(des1, des2, k=2)

        good = []
        for pair in matches_knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 80:
            print(f"[WARN] Frame {i}: too few matches ({len(good)}), skipping")
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
            print(f"[WARN] Frame {i}: findEssentialMat failed, skipping")
            kp1, des1 = kp2, des2
            prev = img
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # Relative transform (direction only; scale unknown)
        T_rel = np.eye(4, dtype=np.float64)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t[:, 0]

        # Accumulate pose: world_T_cam
        T = T @ np.linalg.inv(T_rel)
        traj.append(T[:3, 3].copy())

        if i % 100 == 0:
            print(f"[INFO] Processed {i}/{len(img_paths)-1}")

        kp1, des1 = kp2, des2
        prev = img

    traj = np.array(traj)
    np.save(os.path.join(out_dir, "trajectory.npy"), traj)

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 2])
    plt.xlabel("X (arb. units)")
    plt.ylabel("Z (arb. units)")
    plt.title("Monocular VO Trajectory (Scale Unknown)")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "trajectory.png"), dpi=200)
    plt.show()

    print(f"[DONE] Saved: {os.path.join(out_dir, 'trajectory.png')}")


if __name__ == "__main__":
    main()
