#include "sift_feature_tracker.h"

#include <glog/logging.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;

SiftFeatureTracker::SiftFeatureTracker() : FeatureTracker(), detector(SIFT::create()) {}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  DELIVERABLE 4 | Feature Descriptors (SIFT)
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// Complete the `detectKeypoints` and `describeKeypoints`

/** Detect keypoints in an image. */
void SiftFeatureTracker::detectKeypoints(const cv::Mat& img,
                                         std::vector<KeyPoint>* keypoints) const {
  CHECK_NOTNULL(keypoints);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ~~~~ begin solution
  // Use the SIFT detector to detect keypoints in the image
  detector->detect(img, *keypoints);
  // ~~~~ end solution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

/** Describe keypoints in an image. */
void SiftFeatureTracker::describeKeypoints(const cv::Mat& img,
                                           std::vector<KeyPoint>* keypoints,
                                           cv::Mat* descriptors) const {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ~~~~ begin solution
  // Use the SIFT detector to compute descriptors for the keypoints
  detector->compute(img, *keypoints, *descriptors);
  // ~~~~ end solution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  DELIVERABLE 5 | Feature Descriptors (SIFT)
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
// Complete the matchDescriptors function

void SiftFeatureTracker::matchDescriptors(const cv::Mat& descriptors_1,
                                          const cv::Mat& descriptors_2,
                                          std::vector<std::vector<DMatch>>* matches,
                                          std::vector<cv::DMatch>* good_matches) const {
  CHECK_NOTNULL(matches);
  CHECK_NOTNULL(good_matches);

  // Here we initialize a cv::FlannBasedMatcher for you
  FlannBasedMatcher matcher;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // Match the descriptor vectors using FLANN (See Deliverable 4). Specifically:
  //
  //   1. Take the best 2 using the function FLANN function knnMatch, which
  //   takes the best k nearest neighbours.
  //   Store your matches in the argument 'matches'.
  //
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //
  // ~~~~ begin solution
  matcher.knnMatch(descriptors_1, descriptors_2, *matches, 2);
  // ~~~~ end solution

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   2. Remove ambiguous matches, using SIFT's authors approach (detailed in
  //   the handout). Make use of the DMatch structure.

  // ~~~~ begin solution
  good_matches->clear();
  const float ratio_thresh = 0.75f;  // Lowe’s ratio test threshold
  for (size_t i = 0; i < matches->size(); i++) {
    if ((*matches)[i].size() >= 2) {
      const DMatch& m1 = (*matches)[i][0];
      const DMatch& m2 = (*matches)[i][1];
      if (m1.distance < ratio_thresh * m2.distance) {
        good_matches->push_back(m1);
      }
    }
  }
  // ~~~~ end solution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}
