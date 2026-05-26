# Monocular Visual Odometry + Sparse 3D (MiniCity ROS2 Bag)

This repo contains a simple **monocular Visual Odometry (VO)** pipeline (2D trajectory) and an optional **sparse 3D point cloud** visualization from a **ROS2 bag** (MiniCity dataset).

---

## Prerequisites

### OS / Tools
- Ubuntu (WSL2 is OK) + Bash
- **ROS2 Jazzy** installed and sourced
- `rosbag2` available (`ros2 bag ...`)
- Python **3.12+**
- `git`

### Python packages (inside a virtualenv)
- `opencv-python`
- `numpy`
- `matplotlib`
- `pyyaml`
- (optional for ROS image extraction) `rclpy`, `cv_bridge` (these come with ROS2 install)

---

## Paths used in our local setup (edit if yours differ)

> These paths match **our** machine layout exactly.

- Bag folder (example):  
  `~/bags/minicity_3lap_visual`  *(your bag directory may differ; see NOTE below)*

- Extracted dataset root:  
  `/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/minicity_3lap_visual/minicity_3lap_visual/extracted/`

- Images directory:  
  `/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/minicity_3lap_visual/minicity_3lap_visual/extracted/images/`

- Camera intrinsics YAML:  
  `/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/minicity_3lap_visual/minicity_3lap_visual/extracted/meta/camera_info.yaml`

- VO code directory:  
  `/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/monocular_vo/src/`

- Output directory (generated):  
  `/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/monocular_vo/output/`

**NOTE about bag path:** your bag is wherever you recorded / downloaded it. The README shows the commands; just replace the bag folder accordingly.

---

## 1) ROS2 environment setup (every new terminal)

Run this in **each** terminal you use for ROS2 work:

```bash
source /opt/ros/jazzy/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=0
```

---

## 2) Play the bag (Terminal A)

Go to the bag directory and play it with simulated time:

```bash
cd ~/bags/minicity_3lap_visual   # <-- change to your bag folder
ros2 bag play . --clock -l -r 0.3
```

Keep Terminal A running while you extract metadata/images.

**Quick sanity check that topics exist:**
```bash
ros2 topic list | grep camera
ros2 topic list | grep clock
```

---

## 3) Extract camera intrinsics (Terminal B)

Create the meta folder and dump one `CameraInfo` message to YAML.

First, confirm the camera_info topic name:

```bash
ros2 topic list | grep camera_info
```

Common candidates:
- `/camera/camera/color/camera_info`
- `/camera/color/camera_info`
- `/camera/rgb/camera_info`

Then run (example uses `/camera/camera/color/camera_info`):

```bash
mkdir -p "/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/minicity_3lap_visual/minicity_3lap_visual/extracted/meta"

ros2 topic echo --once /camera/camera/color/camera_info \
  > "/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/minicity_3lap_visual/minicity_3lap_visual/extracted/meta/camera_info.yaml"
```

If you get: **"Could not determine the type for the passed topic"**, it means either:
- the bag is not playing, or
- the topic name is wrong, or
- ROS_DOMAIN_ID / discovery mismatch.

Fix the topic name and retry.

---

## 4) Extract images from the bag (Terminal B)





**Create** `save_images_ros2.py`:
```bash
cd /mnt/c/Users/seris/Downloads/SEM3/ComputerVision


Run the saver (adjust `topic:=...` if needed):
```bash
python3 save_images_ros2.py \
  --ros-args \
  -p topic:=/camera/camera/color/image_raw \
  -p out_dir:=/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/minicity_3lap_visual/minicity_3lap_visual/extracted/images
```

Stop it anytime with `Ctrl+C` once you have enough frames.



---

## 5) Python virtual environment setup (WSL-safe)



Create venv at:
`/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/vo_env`

```bash
cd /mnt/c/Users/seris/Downloads/SEM3/ComputerVision

source vo_env/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install opencv-python numpy matplotlib pyyaml
```

**Activate later (from anywhere):**
```bash
source /mnt/c/Users/seris/Downloads/SEM3/ComputerVision/vo_env/bin/activate
```


---

## 6) Run Basic Monocular VO (trajectory only)

Activate venv, then run `vo.py`.

```bash
 cd /mnt/c/Users/seris/Downloads/SEM3/ComputerVision/monocular_vo/src
 python3 vo.py
```

Expected behavior:
- prints loaded frame count
- prints intrinsic matrix `K`
- shows a 2D trajectory plot (scale unknown / arbitrary)

---

## 7) Run Robust VO

This version reduces skipped frames and logs match statistics.

```bash
 cd /mnt/c/Users/seris/Downloads/SEM3/ComputerVision/monocular_vo/src
 python3 vo_robust.py
```

Expected terminal output includes:
- `good=...` = good feature matches
- `inliers=...` = inliers after Essential matrix RANSAC
- `Skipped frames: 0` (in our run)

---

## 8) Run Sparse 3D Map (triangulated point cloud)

This creates a sparse 3D point cloud (monocular scale is arbitrary).

```bash
source /mnt/c/Users/seris/Downloads/SEM3/ComputerVision/vo_env/bin/activate
cd /mnt/c/Users/seris/Downloads/SEM3/ComputerVision/monocular_vo/src
python3 vo_sparse3d.py

```

### Sanity cleanup (presentation-quality visualization)
We filtered obvious garbage points before plotting:
- remove points with `Z <= 0`
- keep only points with `Z < percentile(95)`
- optionally clamp `|X|, |Y|` percentiles too

This does **not** change VO; it only improves the plot.

---

## 9) Outputs (what you should expect)

All outputs are written here:
`/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/monocular_vo/output/`

Typical files:
- `trajectory.png`  
  2D VO trajectory (scale unknown).
- `trajectory.npy`  
  Saved trajectory array.
- `vo_stats.csv`  
  Per-frame stats (good matches, inliers, etc.).
- `matches_vs_time.png`  
  Plot of tracking quality over time (good matches vs inliers).
- `matches_viz/`  
  Visualization snapshots of matches at sample frames.
- `sparse_points.npy`  
  Saved 3D points (arbitrary scale).
- `sparse_points.png`  
  3D scatter plot of sparse points.

---



---

## Authors
- Surya Prabhav Gurram
- SaiVenkat Reddy Seri
- Soumith Reddy Asani
