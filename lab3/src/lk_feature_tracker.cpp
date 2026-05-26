#include "lk_feature_tracker.h"

#include <algorithm>
#include <cstdio>

// ----------------------------
// Ctor / Dtor
// ----------------------------
LKFeatureTracker::LKFeatureTracker() {}
LKFeatureTracker::~LKFeatureTracker() {}

// ----------------------------
// Helpers
// ----------------------------
void LKFeatureTracker::detectHarrisCorners(const cv::Mat& gray,
                                           std::vector<cv::Point2f>* pts) const {
  std::vector<cv::Point2f> tmp;
  cv::goodFeaturesToTrack(gray,
                          tmp,
                          max_corners_,         // maxCorners
                          quality_,             // qualityLevel
                          min_dist_,            // minDistance
                          cv::noArray(),        // mask
                          block_size_,          // blockSize
                          use_harris_,          // useHarrisDetector
                          harris_k_);           // k

  if (!tmp.empty()) {
    cv::cornerSubPix(gray,
                     tmp,
                     cv::Size(5, 5),           // winSize
                     cv::Size(-1, -1),         // zeroZone
                     cv::TermCriteria(cv::TermCriteria::EPS |
                                       cv::TermCriteria::COUNT,
                                       20, 0.01));
  }

  *pts = std::move(tmp);
}

void LKFeatureTracker::refreshTracks(const std::vector<uchar>& keep_mask) {
  std::vector<std::deque<cv::Point2f>> new_tracks;
  new_tracks.reserve(tracks_.size());

  size_t idx = 0;
  for (size_t i = 0; i < tracks_.size() && idx < curr_pts_.size(); ++i) {
    if (keep_mask[idx]) {
      auto t = tracks_[i];
      t.push_back(curr_pts_[idx]);
      if (t.size() > history_len_) t.pop_front();
      new_tracks.push_back(std::move(t));
    }
    ++idx;
  }
  tracks_.swap(new_tracks);
}

bool LKFeatureTracker::inlierMaskComputation(const std::vector<cv::Point2f>& pts1,
                                             const std::vector<cv::Point2f>& pts2,
                                             std::vector<uchar>* inlier_mask) {
  if (pts1.size() < 8 || pts2.size() < 8) return false;

  // Planar gating is fine for our demo; F-matrix also acceptable
  cv::Mat mask = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0);
  if (mask.empty()) return false;

  inlier_mask->assign(mask.begin<uchar>(), mask.end<uchar>());
  return true;
}

// ----------------------------
// Core: Track one new frame
// ----------------------------
void LKFeatureTracker::trackFeatures(const cv::Mat& frame) {
  CV_Assert(!frame.empty());

  // Ensure we work in grayscale
  cv::Mat gray;
  if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  else gray = frame;

  // First frame: detect, seed tracks, store gray
  if (prev_gray_.empty()) {
    detectHarrisCorners(gray, &prev_pts_);
    tracks_.clear();
    tracks_.reserve(prev_pts_.size());
    for (const auto& p : prev_pts_) {
      std::deque<cv::Point2f> t;
      t.push_back(p);
      tracks_.push_back(std::move(t));
    }
    prev_gray_ = gray.clone();
    return;
  }

  // If input size changed since last call, reset (treat like first frame)
  if (prev_gray_.size() != gray.size()) {
    prev_pts_.clear();
    tracks_.clear();
    detectHarrisCorners(gray, &prev_pts_);
    tracks_.reserve(prev_pts_.size());
    for (const auto& p : prev_pts_) {
      std::deque<cv::Point2f> t;
      t.push_back(p);
      tracks_.push_back(std::move(t));
    }
    prev_gray_ = gray.clone();
    return;
  }

  // If we somehow lost all points, (re)seed before attempting LK
  if (prev_pts_.empty()) {
    detectHarrisCorners(gray, &prev_pts_);
    tracks_.clear();
    tracks_.reserve(prev_pts_.size());
    for (const auto& p : prev_pts_) {
      std::deque<cv::Point2f> t;
      t.push_back(p);
      tracks_.push_back(std::move(t));
    }
    prev_gray_ = gray.clone();
    return;
  }

  // Run pyramidal LK from prev_gray_ -> gray
  curr_pts_.resize(prev_pts_.size());
  cv::calcOpticalFlowPyrLK(prev_gray_, gray,
                           prev_pts_, curr_pts_,
                           status_, err_,
                           win_size_, max_level_, termcrit_,
                           0 /* flags */, 1e-4 /* minEigThreshold */);

  // Optionally gate with geometric inliers
  std::vector<uchar> inlier_mask;
  const bool have_inliers = inlierMaskComputation(prev_pts_, curr_pts_, &inlier_mask);

  // Build keep mask = LK status AND (if available) geometric inlier
  std::vector<uchar> keep(status_.size(), 0);
  for (size_t i = 0; i < status_.size(); ++i) {
    if (!status_[i]) continue;
    if (have_inliers && !inlier_mask[i]) continue;
    keep[i] = 1;
  }

  // Compact points & update histories
  std::vector<cv::Point2f> next_prev; next_prev.reserve(curr_pts_.size());
  size_t write = 0;
  for (size_t i = 0; i < curr_pts_.size(); ++i) {
    if (!keep[i]) continue;
    next_prev.push_back(curr_pts_[i]);

    if (write < tracks_.size()) {
      tracks_[write].push_back(curr_pts_[i]);
      if (tracks_[write].size() > history_len_) tracks_[write].pop_front();
    } else {
      std::deque<cv::Point2f> t; t.push_back(curr_pts_[i]); tracks_.push_back(std::move(t));
    }
    ++write;
  }
  tracks_.resize(write);
  prev_pts_.swap(next_prev);

  // Top up if too few points remain
  if (prev_pts_.size() < static_cast<size_t>(0.25 * max_corners_)) {
    std::vector<cv::Point2f> new_pts; detectHarrisCorners(gray, &new_pts);
    for (const auto& p : new_pts) { prev_pts_.push_back(p); std::deque<cv::Point2f> t; t.push_back(p); tracks_.push_back(std::move(t)); }
  }

  // Advance time
  prev_gray_ = gray.clone();
}

// ----------------------------
// Visualization
// ----------------------------
void LKFeatureTracker::show(const cv::Mat& frame, bool show_window) const {
  cv::Mat vis = render(frame);
  if (show_window) {
    cv::imshow("LK", vis);
    cv::waitKey(1);
  }
}

cv::Mat LKFeatureTracker::render(const cv::Mat& frame) const {
  if (frame.empty()) return frame.clone();
  cv::Mat vis = frame.clone();
  int track_count = 0;

  for (const auto& t : tracks_) {
    if (t.size() < 2) continue;
    ++track_count;
    for (size_t i = 1; i < t.size(); ++i) {
      cv::line(vis, t[i - 1], t[i], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    cv::circle(vis, t.back(), 3, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
  }

  cv::putText(vis,
              "track count: " + std::to_string(track_count),
              cv::Point(12, 32),
              cv::FONT_HERSHEY_SIMPLEX, 0.9,
              cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  return vis;
}
