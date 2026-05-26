// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  Fall 2025  - Lab 3 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement an AKAZE feature tracker derived class
//  that inherits from your FeatureTracker base class.
//
// NOTE: Deliverables for the TEAM portion of this assignment start at number 4
// and end at number 11.  Deliverables 1-3 are
// individual.
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  DELIVERABLE 7 | Comparing Feature Matching Algorithms on Real Data
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
// For this part, you will need to implement the same functions you've just
// implemented in the case of SIFT, but now for AKAZE features. You'll also
// implement these functions for the case of ORB features and BRISK. For
// those cases, see orb_feature_tracker.cpp and brisk_feature_tracker.cpp (and
// respective headers)
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include "akaze_feature_tracker.h"

#include <glog/logging.h>

#include <opencv2/features2d.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <vector>

using namespace cv;

AkazeFeatureTracker::AkazeFeatureTracker()
    : FeatureTracker(), detector(AKAZE::create()) {}

/** TODO: this function detects keypoints in an image.
    @param[in] img Image input where to detect keypoints.
    @param[out] keypoints List of keypoints detected on the given image.
*/
void AkazeFeatureTracker::detectKeypoints(const cv::Mat& img,
                                          std::vector<KeyPoint>* keypoints) const {
  CHECK_NOTNULL(keypoints);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ~~~~ begin solution
  // Use the AKAZE detector to detect keypoints
  detector->detect(img, *keypoints);
  // ~~~~ end solution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

void AkazeFeatureTracker::describeKeypoints(const cv::Mat& img,
                                            std::vector<KeyPoint>* keypoints,
                                            cv::Mat* descriptors) const {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ~~~~ begin solution
  // Use the AKAZE detector to compute descriptors for the keypoints
  detector->compute(img, *keypoints, *descriptors);
  // ~~~~ end solution
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
}

void AkazeFeatureTracker::matchDescriptors(
    const cv::Mat& descriptors_1,
    const cv::Mat& descriptors_2,
    std::vector<std::vector<DMatch>>* matches,
    std::vector<cv::DMatch>* good_matches) const {
  CHECK_NOTNULL(matches);

  std::vector<std::vector<DMatch>> backward_matches;

  // Here we initialize a FlannBasedMatcher for you
  FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // This should be exactly the same as what you wrote for the SIFT feature
  // tracker.
  //
  // ~~~~ begin solution
  matcher.knnMatch(descriptors_1, descriptors_2, *matches, 2);

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
