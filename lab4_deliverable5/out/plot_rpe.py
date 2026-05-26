#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import statistics as stats
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ------------------------------
# CSV loading (flexible headers)
# ------------------------------
def _pick(headers, *cands):
    for c in cands:
        if c in headers:
            return c
    raise KeyError(f"Missing columns; need one of {cands}. Found: {headers}")

def load_csv_generic(path: str) -> Tuple[List[int], List[float], List[float]]:
    """
    Return (frames, rot_deg, trans_dir_err) from a CSV with flexible headers.
    Expected columns (any one of each group):
      frame: ('frame', 'frame_idx', 'idx', 'step', 't')
      rot:   ('rot_err_deg', 'rotation_error_deg', 'rot_error_deg', 'rot_deg', 'rot')
      trans: ('trans_dir_err', 'transl_dir_err', 'translation_dir_err', 'translation_error_dir',
              'trans_err_dir', 'u_err', 't_err_dir', 't_err', 'trans_err')
    """
    frames, r_err, t_err = [], [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        k_frame = _pick(headers, "frame", "frame_idx", "idx", "step", "t")
        k_rot   = _pick(headers, "rot_err_deg", "rotation_error_deg", "rot_error_deg", "rot_deg", "rot")
        k_trans = _pick(headers, "trans_dir_err", "transl_dir_err", "translation_dir_err",
                        "translation_error_dir", "trans_err_dir", "u_err", "t_err_dir", "t_err", "trans_err")
        for row in reader:
            try:
                frames.append(int(float(row[k_frame])))
            except Exception:
                # fallback if non-integer frame indices
                frames.append(len(frames))
            # rotation (deg)
            try:
                r_err.append(float(row[k_rot]))
            except Exception:
                r_err.append(float("nan"))
            # translation direction error
            try:
                t_err.append(float(row[k_trans]))
            except Exception:
                t_err.append(float("nan"))
    return frames, r_err, t_err


# ------------------------------
# Plot helpers
# ------------------------------
def _series_stats(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return stats.mean(vals), stats.median(vals)

def _plot_series(ax, x, y, label: str):
    ax.plot(x, y, label=label, linewidth=1.6)

def _finalize_plot(ax, title: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv5", required=True, help="CSV for 5-point")
    ap.add_argument("--csv8", required=True, help="CSV for 8-point (or reuse for Arun/3pt)")
    ap.add_argument("--csv2", required=True, help="CSV for 2-point (known R)")
    # NEW: optional 3-point (Arun)
    ap.add_argument("--csv3", required=False, help="CSV for 3-point (Arun)")

    ap.add_argument("--outprefix", default="rpe_plot", help="Output file prefix")

    ap.add_argument("--label5", default="5-point (Nister, RANSAC)")
    ap.add_argument("--label8", default="8-point (Longuet-Higgins, RANSAC)")
    ap.add_argument("--label2", default="2-point (known R, RANSAC)")
    # NEW: label for Arun
    ap.add_argument("--label3", default="3-point (Arun)")

    args = ap.parse_args()

    # Load mandatory series
    f5, r5, t5 = load_csv_generic(args.csv5)
    f8, r8, t8 = load_csv_generic(args.csv8)
    f2, r2, t2 = load_csv_generic(args.csv2)

    # Optionally load Arun
    f3: Optional[List[int]] = None
    r3: Optional[List[float]] = None
    t3: Optional[List[float]] = None
    if args.csv3:
        f3, r3, t3 = load_csv_generic(args.csv3)

    # --- Print stats ---
    m5, md5 = _series_stats(r5); mt5, mdt5 = _series_stats(t5)
    m8, md8 = _series_stats(r8); mt8, mdt8 = _series_stats(t8)
    m2, md2 = _series_stats(r2); mt2, mdt2 = _series_stats(t2)

    print("\n=== Rotation error (deg) ===")
    print(f"{args.label5:30s}  mean={m5:.3f}  median={md5:.3f}")
    print(f"{args.label8:30s}  mean={m8:.3f}  median={md8:.3f}")
    print(f"{args.label2:30s}  mean={m2:.3f}  median={md2:.3f}")
    if r3 is not None:
        m3, md3 = _series_stats(r3)
        print(f"{args.label3:30s}  mean={m3:.3f}  median={md3:.3f}")

    print("\n=== Translation direction error ===")
    print(f"{args.label5:30s}  mean={mt5:.3f}  median={mdt5:.3f}")
    print(f"{args.label8:30s}  mean={mt8:.3f}  median={mdt8:.3f}")
    print(f"{args.label2:30s}  mean={mt2:.3f}  median={mdt2:.3f}")
    if t3 is not None:
        mt3, mdt3 = _series_stats(t3)
        print(f"{args.label3:30s}  mean={mt3:.3f}  median={mdt3:.3f}")

    # --- Plots ---
    # Translation
    fig_t, ax_t = plt.subplots(figsize=(10, 5))
    _plot_series(ax_t, f5, t5, args.label5)
    _plot_series(ax_t, f8, t8, args.label8)
    _plot_series(ax_t, f2, t2, args.label2)
    if t3 is not None:
        _plot_series(ax_t, f3, t3, args.label3)
    _finalize_plot(ax_t, "Relative Pose Error — Translation direction", "direction error (0..2)")
    fig_t.tight_layout()
    fig_t.savefig(f"{args.outprefix}_translation.png", dpi=180)

    # Rotation
    fig_r, ax_r = plt.subplots(figsize=(10, 5))
    _plot_series(ax_r, f5, r5, args.label5)
    _plot_series(ax_r, f8, r8, args.label8)
    _plot_series(ax_r, f2, r2, args.label2)
    if r3 is not None:
        _plot_series(ax_r, f3, r3, args.label3)
    _finalize_plot(ax_r, "Relative Pose Error — Rotation", "degrees")
    fig_r.tight_layout()
    fig_r.savefig(f"{args.outprefix}_rotation.png", dpi=180)

    print(f"\nSaved: {args.outprefix}_translation.png")
    print(f"Saved: {args.outprefix}_rotation.png")


if __name__ == "__main__":
    main()
