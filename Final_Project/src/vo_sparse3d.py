import os
import glob
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt

def load_intrinsics(cam_yaml_path: str) -> np.ndarray:
    # Your camera_info.yaml may contain multiple YAML docs. Take the first one.
    with open(cam_yaml_path, "r") as f:
        docs = list(yaml.safe_load_all(f))
    data = docs[0] if isinstance(docs, list) else docs
    k = data["k"]  # row-major 3x3
    K = np.array(k, dtype=np.float64).reshape(3, 3)
    return K

def load_images(img_dir: str):
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(paths) < 2:
        raise RuntimeError(f"Not enough images in {img_dir}")
    return paths

def orb_match(img1, img2, nfeatures=3000):
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=12,
    )

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 50:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, good

def triangulate_pair(K, R, t, pts1, pts2):
    # Projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    # Triangulate in homogeneous coords (4xN)
    X_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (X_h[:3] / (X_h[3] + 1e-12)).T  # Nx3
    return X

def filter_points(R, t, X, max_depth=50.0):
    # Cheirality check: points should be in front of both cameras.
    # Cam1: Z > 0
    z1 = X[:, 2]

    # Cam2: transform points into cam2 frame: X2 = R*X + t
    X2 = (R @ X.T + t).T
    z2 = X2[:, 2]

    mask = (z1 > 0.1) & (z2 > 0.1)
    Xf = X[mask]

    # Remove extreme depths (usually garbage)
    Xf = Xf[np.abs(Xf[:, 2]) < max_depth]
    Xf = Xf[np.abs(Xf[:, 0]) < max_depth]
    Xf = Xf[np.abs(Xf[:, 1]) < max_depth]
    return Xf

def main():
    base = "/mnt/c/Users/seris/Downloads/SEM3/ComputerVision"
    data_root = os.path.join(base, "minicity_3lap_visual", "minicity_3lap_visual")
    img_dir = os.path.join(data_root, "extracted", "images")
    cam_yaml = os.path.join(data_root, "extracted", "meta", "camera_info.yaml")

    out_dir = os.path.join(base, "monocular_vo", "output")
    os.makedirs(out_dir, exist_ok=True)

    K = load_intrinsics(cam_yaml)
    img_paths = load_images(img_dir)
    print(f"[INFO] Loaded {len(img_paths)} images")
    print(f"[INFO] K=\n{K}")

    # Accumulate a sparse point cloud across selected frames
    all_points = []

    # Use every Nth frame to avoid massive redundancy
    step = 5  # change to 3 for denser, 10 for faster
    for i in range(0, len(img_paths) - step, step):
        img1 = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_paths[i + step], cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue

        m = orb_match(img1, img2)
        if m is None:
            continue
        pts1, pts2, good = m

        # Estimate Essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        if E is None or mask is None:
            continue

        inlier_mask = mask.ravel().astype(bool)
        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]
        if len(pts1_in) < 50:
            continue

        # Recover pose
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)
        if R is None or t is None:
            continue

        # Triangulate using inlier correspondences
        X = triangulate_pair(K, R, t, pts1_in, pts2_in)
        Xf = filter_points(R, t, X, max_depth=60.0)

        if Xf.shape[0] > 0:
            all_points.append(Xf)

        if (i // step) % 50 == 0:
            total = sum(p.shape[0] for p in all_points) if all_points else 0
            print(f"[INFO] Frame {i}: inliers={len(pts1_in)} | added={Xf.shape[0]} | total_pts={total}")

    if not all_points:
        print("[FAIL] No points triangulated. Drop this option.")
        return

    P = np.vstack(all_points)

    # Subsample for plotting so it doesn't freeze
    P = np.vstack(all_points)

    # ---------- Visualization-only cleanup ----------
    # 1) Remove points behind the camera
    P = P[P[:, 2] > 0]

    # 2) Remove extreme depth outliers (top 5%)
    z_max = np.percentile(P[:, 2], 95)
    P = P[P[:, 2] < z_max]

    # 3) Optional: clip X/Y outliers for cleaner visualization
    x_lim = np.percentile(np.abs(P[:, 0]), 95)
    y_lim = np.percentile(np.abs(P[:, 1]), 95)
    P = P[(np.abs(P[:, 0]) < x_lim) & (np.abs(P[:, 1]) < y_lim)]

    print(f"[INFO] After filtering: {P.shape[0]} points")

    # Subsample for plotting
    if P.shape[0] > 20000:
        idx = np.random.choice(P.shape[0], 20000, replace=False)
        P_plot = P[idx]
    else:
        P_plot = P
    # -----------------------------------------------

   

    np.save(os.path.join(out_dir, "sparse_points.npy"), P)
    print(f"[DONE] Saved points: {P.shape[0]} -> {os.path.join(out_dir, 'sparse_points.npy')}")

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P_plot[:, 0], P_plot[:, 1], P_plot[:, 2], s=1)
    ax.set_title("Sparse 3D Points (Monocular, Scale Arbitrary)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    out_png = os.path.join(out_dir, "sparse_points.png")
    plt.savefig(out_png, dpi=200)
    print(f"[DONE] Saved plot: {out_png}")
    plt.show()

if __name__ == "__main__":
    main()
