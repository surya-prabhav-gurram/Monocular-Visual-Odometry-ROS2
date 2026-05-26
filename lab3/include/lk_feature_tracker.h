#pragma once
#include <opencv2/opencv.hpp>
#include <deque>

/**
 * Harris + Lucas–Kanade feature tracker.
 * - Detects Harris corners on the first frame
 * - Tracks with cv::calcOpticalFlowPyrLK
 * - Optional geometric gating (RANSAC homography)
 * - Maintains short track histories for visualization
 */
class LKFeatureTracker {
public:
  LKFeatureTracker();
  ~LKFeatureTracker();

  // Consume one RGB/BGR/GRAY frame and update tracks
  void trackFeatures(const cv::Mat& frame);

  // Draw tracks on a copy of 'frame' and display (non-blocking)
  void show(const cv::Mat& frame, bool show_window = true) const;

  // Return a copy of 'frame' with tracks drawn (no window)
  cv::Mat render(const cv::Mat& frame) const;

  // Compute a robust inlier mask between two 2D point sets (RANSAC)
  static bool inlierMaskComputation(const std::vector<cv::Point2f>& pts1,
                                    const std::vector<cv::Point2f>& pts2,
                                    std::vector<uchar>* inlier_mask);

private:
  // Harris params (tune for density/sensitivity)
  int    max_corners_   = 1200;
  double quality_       = 0.005;
  double min_dist_      = 5.0;
  int    block_size_    = 3;
  bool   use_harris_    = true;
  double harris_k_      = 0.04;

  // LK params
  cv::Size win_size_    = cv::Size(31,31);
  int      max_level_   = 4;
  cv::TermCriteria termcrit_ =
      cv::TermCriteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS, 30, 0.03);

  // State
  cv::Mat prev_gray_;
  std::vector<cv::Point2f> prev_pts_, curr_pts_;
  std::vector<uchar> status_;
  std::vector<float> err_;

  // Track history
  std::vector<std::deque<cv::Point2f>> tracks_;
  size_t history_len_ = 20;

  // Helpers
  void detectHarrisCorners(const cv::Mat& gray, std::vector<cv::Point2f>* pts) const;
  void refreshTracks(const std::vector<uchar>& keep_mask);
};
