// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//   Fall 2025  - Lab 3 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement a generic feature tracker base class
//  which you will augment with several derived classes for SIFT, AKAZE, ORB,
//  and BRISK feature tracking.
//
// NOTE: Deliverables for the TEAM portion of this assignment start at number 4
// and end at number 11.  Deliverables 1-3 are
// individual.

#include "feature_tracker.h"

#include <glog/logging.h>

#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using namespace cv;

/** (TODO) This is the main tracking function, given two images, it detects,
 * describes and matches features.
 * We will be modifying this function incrementally to plot different figures
 * and compute different statistics.
  @param[in] img_1, img_2 Images where to track features.
  @param[out] matched_kp_1_kp_2 pair of vectors of keypoints with the same size
  so that matched_kp_1_kp_2.first[i] matches with matched_kp_1_kp_2.second[i].
*/
void FeatureTracker::trackFeatures(
    const cv::Mat& img_1,
    const cv::Mat& img_2,
    std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>>* matched_kp_1_kp_2,
    const bool save_images,
    const bool show_images) {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  DELIVERABLE 4 | Feature Descriptors (SIFT)
  std::vector<KeyPoint> keypoints_1, keypoints_2;

  // 1. Detect keypoints
  detectKeypoints(img_1, &keypoints_1);
  detectKeypoints(img_2, &keypoints_2);

  // 2. Optionally draw and save detected keypoints
  if (save_images || show_images) {
    cv::Mat img1_kp, img2_kp;
    cv::drawKeypoints(img_1, keypoints_1, img1_kp);
    cv::drawKeypoints(img_2, keypoints_2, img2_kp);
    if (save_images) {
      cv::imwrite("keypoints_img1.png", img1_kp);
      cv::imwrite("keypoints_img2.png", img2_kp);
    }
    if (show_images) {
      cv::imshow("Keypoints Image1", img1_kp);
      cv::imshow("Keypoints Image2", img2_kp);
      cv::waitKey(1);
    }
  }

  // 3. Compute descriptors for both images
  cv::Mat descriptors_1, descriptors_2;
  describeKeypoints(img_1, &keypoints_1, &descriptors_1);
  describeKeypoints(img_2, &keypoints_2, &descriptors_2);

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  DELIVERABLE 5 | Descriptor-based Feature Matching
  std::vector<std::vector<DMatch>> matches;
  std::vector<DMatch> good_matches;

  // 1. Match descriptor vectors using FLANN matcher
  matchDescriptors(descriptors_1, descriptors_2, &matches, &good_matches);

  // 2. Plot the good matches using the opencv function 'drawMatches'. Save image.
  if (save_images || show_images) {
    cv::Mat img_matches;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2,
                    good_matches, img_matches,
                    Scalar::all(-1), Scalar::all(-1),
                    std::vector<char>(),
                    DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    if (save_images) {
      cv::imwrite("good_matches.png", img_matches);
    }
    if (show_images) {
      cv::imshow("Good Matches", img_matches);
      cv::waitKey(1);
    }
  }
  std::vector<cv::DMatch> flat_matches;
for (const auto &vec : matches) {
    if (!vec.empty()) {
        flat_matches.push_back(vec[0]);  // take first match
    }
}
if (save_images || show_images) {
    cv::Mat img_allmatches;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2,
                    flat_matches, img_allmatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    if (save_images) {
        cv::imwrite("all_matches.png", img_allmatches);
    }
    if (show_images) {
        cv::imshow("All Matches (raw)", img_allmatches);
        cv::waitKey(1);
    }
}

  // 3. Build aligned keypoints for matches (needed for inlier mask)
  std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> match_kp_1_kp_2_local;
  for (const auto& m : good_matches) {
    match_kp_1_kp_2_local.first.push_back(keypoints_1[m.queryIdx]);
    match_kp_1_kp_2_local.second.push_back(keypoints_2[m.trainIdx]);
  }
  // ------------------------------------------------------
// DELIVERABLE 6 | Inlier vs Outlier Matches
// ------------------------------------------------------

// 1. Compute inlier mask
std::vector<uchar> inlier_mask;
inlierMaskComputation(match_kp_1_kp_2_local.first,
                      match_kp_1_kp_2_local.second,
                      &inlier_mask);

// 2. Count inliers
unsigned int num_inliers = 0;
for (uchar mask_val : inlier_mask) {
    if (mask_val) num_inliers++;
}

// 3. Draw outliers first (default color), then inliers in green
if (save_images || show_images) {
    // Make a copy of good_matches but with outliers only
    std::vector<cv::DMatch> outlier_matches;
    for (size_t i = 0; i < good_matches.size(); ++i) {
        if (!inlier_mask[i]) {
            outlier_matches.push_back(good_matches[i]);
        }
    }

    // Draw outliers first (in red)
    cv::Mat img_outliers;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2,
                    outlier_matches, img_outliers,
                    cv::Scalar(0, 0, 255), // red lines
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Draw inliers on top (in green)
    std::vector<cv::DMatch> inlier_matches;
    for (size_t i = 0; i < good_matches.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_matches.push_back(good_matches[i]);
        }
    }

    // <<< REPLACE THIS OLD drawMatches call here >>>
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2,
                    inlier_matches, img_outliers,
                    cv::Scalar(0, 255, 0), // green lines
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::DRAW_OVER_OUTIMG |
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if (save_images) {
        cv::imwrite("inlier_outlier_matches.png", img_outliers);
    }
    if (show_images) {
        cv::imshow("Inlier vs Outlier Matches", img_outliers);
        cv::waitKey(1);
    }
}



  // keep the rest of the original code unchanged:
  


  double new_num_samples = static_cast<double>(num_samples_) + 1.0f;
  double old_num_samples = static_cast<double>(num_samples_);
  avg_num_keypoints_img1_ = (avg_num_keypoints_img1_ * old_num_samples +
                             static_cast<double>(keypoints_1.size())) /
                            new_num_samples;
  avg_num_keypoints_img2_ = (avg_num_keypoints_img2_ * old_num_samples +
                             static_cast<double>(keypoints_2.size())) /
                            new_num_samples;
  avg_num_matches_ =
      (avg_num_matches_ * old_num_samples + static_cast<double>(matches.size())) /
      new_num_samples;
  avg_num_good_matches_ = (avg_num_good_matches_ * old_num_samples +
                           static_cast<double>(good_matches.size())) /
                          new_num_samples;
  avg_num_inliers_ =
      (avg_num_inliers_ * old_num_samples + static_cast<double>(num_inliers)) /
      new_num_samples;
  avg_inlier_ratio_ =
      (avg_inlier_ratio_ * old_num_samples +
       (static_cast<double>(num_inliers) / static_cast<double>(good_matches.size()))) /
      new_num_samples;
  ++num_samples_;

  if (matched_kp_1_kp_2 != nullptr) {
    *matched_kp_1_kp_2 = match_kp_1_kp_2_local;
  }
  printStats();
}

void FeatureTracker::printStats() const {
  LOG(INFO) << "Avg. Keypoints 1 Size: " << avg_num_keypoints_img1_;
  LOG(INFO) << "Avg. Keypoints 2 Size: " << avg_num_keypoints_img2_;
  LOG(INFO) << "Avg. Number of matches: " << avg_num_matches_;
  LOG(INFO) << "Avg. Number of good matches: " << avg_num_good_matches_;
  LOG(INFO) << "Avg. Number of Inliers: " << avg_num_inliers_;
  LOG(INFO) << "Avg. Inliers ratio: " << avg_inlier_ratio_;
  LOG(INFO) << "Num. of samples: " << num_samples_;
}

FeatureTracker::~FeatureTracker() { printStats(); }

void FeatureTracker::inlierMaskComputation(const std::vector<KeyPoint>& keypoints_1,
                                           const std::vector<KeyPoint>& keypoints_2,
                                           std::vector<uchar>* inlier_mask) const {
  CHECK_NOTNULL(inlier_mask);
  const size_t size = keypoints_1.size();
  CHECK_EQ(keypoints_2.size(), size) << "Size of keypoint vectors "
                                        "should be the same!";

  std::vector<Point2f> pts1(size);
  std::vector<Point2f> pts2(size);
  for (size_t i = 0; i < keypoints_1.size(); i++) {
    pts1[i] = keypoints_1[i].pt;
    pts2[i] = keypoints_2[i].pt;
  }

  static constexpr double max_dist_from_epi_line_in_px = 3.0;
  static constexpr double confidence_prob = 0.99;
  try {
    findFundamentalMat(pts1,
                       pts2,
                       FM_RANSAC,
                       max_dist_from_epi_line_in_px,
                       confidence_prob,
                       *inlier_mask);
  } catch (...) {
    LOG(WARNING) << "Inlier Mask could not be computed, this can happen if "
                    "there are not enough features tracked.";
  }
}

void FeatureTracker::drawMatches(const cv::Mat& img_1,
                                 const cv::Mat& img_2,
                                 const std::vector<KeyPoint>& keypoints_1,
                                 const std::vector<KeyPoint>& keypoints_2,
                                 const std::vector<std::vector<DMatch>>& matches,
                                 const bool show_images) {
  cv::namedWindow("tracked_features", cv::WINDOW_NORMAL);
  cv::Mat img_matches;
  cv::drawMatches(img_1,
                  keypoints_1,
                  img_2,
                  keypoints_2,
                  matches,
                  img_matches,
                  Scalar::all(-1),
                  Scalar::all(-1),
                  std::vector<std::vector<char>>(),
                  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  if (show_images) {
    imshow("tracked_features", img_matches);
  }
}
