/*
 * @file pose_estimation.cpp
 * @brief Estimates the pose from frame to frame.
 */
#include <cv_bridge/cv_bridge.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logging.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <Eigen/Eigen>
#include <iostream>
#include <memory>
#include <functional>
#include <fstream>
#include <stdio.h>
#include <map>   // CSV bookkeeping
#include <random>  // <-- needed by your 3pt RANSAC

// Feature tracker headers (lab 3)
#include "lab_3/akaze_feature_tracker.h"
#include "lab_3/brisk_feature_tracker.h"
#include "lab_3/feature_tracker.h"
#include "lab_3/orb_feature_tracker.h"
#include "lab_3/sift_feature_tracker.h"

// OpenGV
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/point_cloud/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>

// Lab 4 utilities
#include "lab4_utils.h"
#include "pose_estimation.h"

// Type aliases
using Adapter  = opengv::relative_pose::CentralRelativeAdapter;
using RansacProblem =
    opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;

using AdapterGivenRot = opengv::relative_pose::CentralRelativeAdapter;
using RansacProblemGivenRot =
    opengv::sac_problems::relative_pose::TranslationOnlySacProblem;

using Adapter3D = opengv::point_cloud::PointCloudAdapter;
using RansacProblem3D = opengv::sac_problems::point_cloud::PointCloudSacProblem;

class PoseEstimator : public rclcpp::Node {
 public:
  bool use_ransac_;
  bool scale_translation_;
  bool show_images_;
  int pose_estimator_;

  // NEW: allow RGB-only mode (no depth)
  bool use_depth_images_;   // if false -> RGB-only subscriber

  std::shared_ptr<image_transport::SubscriberFilter> sf_rgb_;
  std::shared_ptr<image_transport::SubscriberFilter> sf_depth_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>
      MySyncPolicy;
  std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

  std::unique_ptr<FeatureTracker> feature_tracker_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_pose_estimation_, pub_pose_gt_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
  image_transport::Subscriber img_sub_;
  geometry_msgs::msg::PoseStamped curr_pose_;
  geometry_msgs::msg::PoseStamped prev_pose_;

  CameraParams camera_params_;
  cv::Mat R_camera_body, t_camera_body;
  cv::Mat T_camera_body;
  geometry_msgs::msg::Pose pose_camera_body;
  tf2::Transform transform_camera_body;

  PoseEstimator() : Node("pose_estimator") {
    declare_parameter<bool>("use_ransac");
    declare_parameter<bool>("scale_translation");
    declare_parameter<bool>("show_images");
    declare_parameter<int>("pose_estimator");
    declare_parameter<bool>("use_depth_images", false);  // NEW (default false)

    if (!(get_parameter("use_ransac", use_ransac_) &&
          get_parameter("scale_translation", scale_translation_) &&
          get_parameter("pose_estimator", pose_estimator_))) {
      RCLCPP_ERROR(get_logger(),
                   "Must set use_ransac, scale_translation, and pose_estimator params");
      exit(1);
    }
    get_parameter("show_images", show_images_);
    get_parameter("use_depth_images", use_depth_images_);  // NEW

    // intrinsics
    camera_params_.K = cv::Mat::zeros(3, 3, CV_64F);
    camera_params_.K.at<double>(0, 0) = 415.69219381653056;
    camera_params_.K.at<double>(1, 1) = 415.69219381653056;
    camera_params_.K.at<double>(0, 2) = 360.0;
    camera_params_.K.at<double>(1, 2) = 240.0;
    camera_params_.D = cv::Mat::zeros(cv::Size(5, 1), CV_64F);

    // T_camera^body
    T_camera_body = cv::Mat::zeros(cv::Size(4, 4), CV_64F);
    T_camera_body.at<double>(0, 2) = 1.0;
    T_camera_body.at<double>(1, 0) = -1.0;
    T_camera_body.at<double>(1, 3) = 0.05;
    T_camera_body.at<double>(2, 1) = -1.0;
    T_camera_body.at<double>(3, 3) = 1.0;
    R_camera_body = T_camera_body(cv::Range(0, 3), cv::Range(0, 3));
    t_camera_body = T_camera_body(cv::Range(0, 3), cv::Range(3, 4));
    pose_camera_body = cv2Pose(R_camera_body, t_camera_body);
    tf2::convert(pose_camera_body, transform_camera_body);

    feature_tracker_.reset(new SiftFeatureTracker());

    // Ground-truth odom (body), we convert to GT camera pose
    pose_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/ground_truth_pose", 10,
        std::bind(&PoseEstimator::poseCallbackTesse, this, std::placeholders::_1));

    pub_pose_gt_ = create_publisher<geometry_msgs::msg::PoseStamped>("/gt_camera_pose", 1);
    pub_pose_estimation_ = create_publisher<geometry_msgs::msg::PoseStamped>("/camera_pose", 1);
  }

  void run() {
    std::string transport = "raw";
    image_transport::TransportHints hints(this, transport);

    // These topics are remapped by the launch file
    if (use_depth_images_) {
      // Original RGB+Depth synchronized path
      sf_rgb_ = std::make_shared<image_transport::SubscriberFilter>(
          this, "/rgb_images_topic", hints.getTransport());
      sf_depth_ = std::make_shared<image_transport::SubscriberFilter>(
          this, "/depth_images_topic", hints.getTransport());

      sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(
          MySyncPolicy(10), *sf_rgb_, *sf_depth_);
      sync_->registerCallback(std::bind(&PoseEstimator::cameraCallback,
                                        this,
                                        std::placeholders::_1,
                                        std::placeholders::_2));
    } else {
      // NEW: RGB-only subscription (for real bag with color only)
      img_sub_ = image_transport::create_subscription(
          this,
          "/rgb_images_topic",
          std::bind(&PoseEstimator::rgbOnlyCallback, this, std::placeholders::_1),
          hints.getTransport());
      RCLCPP_INFO(get_logger(), "Running in RGB-only mode (use_depth_images:=false)");
    }
  }

  void poseCallbackTesse(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    // GT body pose -> camera pose via fixed extrinsics
    curr_pose_.pose = msg->pose.pose;
    tf2::Transform T_WB;
    tf2::convert(curr_pose_.pose, T_WB);
    tf2::toMsg(T_WB * transform_camera_body, curr_pose_.pose);
    curr_pose_.header.frame_id = "world";
    pub_pose_gt_->publish(curr_pose_);
  }

  // undistort + to bearing vectors (normalized)
  void calibrateKeypoints(const std::vector<cv::Point2f>& pts1,
                          const std::vector<cv::Point2f>& pts2,
                          opengv::bearingVectors_t& bv1,
                          opengv::bearingVectors_t& bv2) {
    std::vector<cv::Point2f> p1u, p2u;
    cv::undistortPoints(pts1, p1u, camera_params_.K, camera_params_.D);
    cv::undistortPoints(pts2, p2u, camera_params_.K, camera_params_.D);
    bv1.clear(); bv2.clear();
    bv1.reserve(p1u.size()); bv2.reserve(p2u.size());
    for (const auto& p : p1u) {
      opengv::bearingVector_t v(p.x, p.y, 1.0);
      bv1.push_back(v.normalized());
    }
    for (const auto& p : p2u) {
      opengv::bearingVector_t v(p.x, p.y, 1.0);
      bv2.push_back(v.normalized());
    }
  }

  void updatePoseEstimate(const geometry_msgs::msg::Pose& prev_pose,
                          const geometry_msgs::msg::Pose& rel_pose,
                          geometry_msgs::msg::Pose& out_pose) {
    tf2::Transform Tprev, Trel;
    tf2::convert(prev_pose, Tprev);
    tf2::convert(rel_pose, Trel);
    tf2::toMsg(Tprev * Trel, out_pose);
  }

  void scaleTranslation(geometry_msgs::msg::Point& t,
                        const geometry_msgs::msg::PoseStamped& prev_pose,
                        const geometry_msgs::msg::PoseStamped& curr_pose) {
    if (!scale_translation_) return;
    tf2::Transform Tprev, Tcurr;
    tf2::convert(prev_pose.pose, Tprev);
    tf2::convert(curr_pose.pose, Tcurr);
    tf2::Transform Trel = Tprev.inverseTimes(Tcurr);
    const double s = Trel.getOrigin().length();
    if (!(std::isfinite(s) && s > 1e-12)) return;
    const double old = std::sqrt(t.x*t.x + t.y*t.y + t.z*t.z);
    if (old < 1e-12) return;
    const double k = s / old;
    t.x *= k; t.y *= k; t.z *= k;
  }

  // ---------- RPE with CSV logging ----------
  void evaluateRPE(const tf2::Transform& gt_prev,
                   const tf2::Transform& gt_curr,
                   const tf2::Transform& est_prev,
                   const tf2::Transform& est_curr) {
    const tf2::Transform est_rel = est_prev.inverseTimes(est_curr);
    const tf2::Transform gt_rel  = gt_prev.inverseTimes(gt_curr);

    // rotation error (deg)
    tf2::Quaternion q_err = gt_rel.getRotation().inverse() * est_rel.getRotation();
    q_err.normalize();
    double ang = 2.0 * std::atan2(std::sqrt(q_err.x()*q_err.x() + q_err.y()*q_err.y() + q_err.z()*q_err.z()),
                                  std::abs(q_err.w()));
    if (!std::isfinite(ang)) ang = 0.0;
    const double rot_err_deg = ang * 180.0 / M_PI;

    // translation *direction* error (0..2)
    const tf2::Vector3 tg = gt_rel.getOrigin();
    const tf2::Vector3 te = est_rel.getOrigin();
    double trans_err = std::numeric_limits<double>::quiet_NaN();
    if (tg.length2() > 1e-12 && te.length2() > 1e-12) {
      trans_err = (tg.normalized() - te.normalized()).length();
    }

    if (std::isfinite(trans_err)) {
      RCLCPP_INFO(this->get_logger(), "RPE  rot=%6.3f deg   trans=%6.3f", rot_err_deg, trans_err);
    } else {
      RCLCPP_WARN(this->get_logger(), "RPE  rot=%6.3f deg   trans=NaN", rot_err_deg);
    }

    // CSV
    static std::map<int,bool> wrote_header;
    static std::map<int,int>  frame_idx;

    const std::string out_dir =
        "/mnt/c/Users/seris/Downloads/SEM3/ComputerVision/TEAM_7/lab4/out";
    std::string fname = out_dir + "/err_unknown.csv";
    if (pose_estimator_ == 0)      fname = out_dir + "/err_5pt.csv";
    else if (pose_estimator_ == 1) fname = out_dir + "/err_8pt.csv";
    else if (pose_estimator_ == 2) fname = out_dir + "/err_2pt.csv";
    else if (pose_estimator_ == 3) fname = out_dir + "/err_3pt.csv";

    std::ofstream f(fname, std::ios::app);
    if (!f.good()) {
      RCLCPP_WARN(this->get_logger(), "Could not open CSV file: %s", fname.c_str());
      return;
    }
    if (!wrote_header[pose_estimator_]) {
      f << "frame_idx,rot_err_deg,trans_err\n";
      wrote_header[pose_estimator_] = true;
      frame_idx[pose_estimator_] = 0;
    }
    const int idx = frame_idx[pose_estimator_]++;
    f << idx << "," << rot_err_deg << ",";
    if (std::isfinite(trans_err)) f << trans_err << "\n"; else f << "nan\n";
    f.close();
  }

  void cameraCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
                      const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg) {
    cv::Mat bgr, depth;
    try {
      bgr   = cv_bridge::toCvShare(rgb_msg, "bgr8")->image;
      depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding)->image;
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "Could not convert rgb or depth images.");
      return;
    }

    static cv::Mat prev_bgr = bgr.clone();
    static cv::Mat prev_depth = depth.clone();

    // Track features
    std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> matched_kp_1_kp_2;
    feature_tracker_->trackFeatures(prev_bgr, bgr, &matched_kp_1_kp_2, false, show_images_);
    std::vector<cv::Point2f> pts1, pts2;
    cv::KeyPoint::convert(matched_kp_1_kp_2.first,  pts1);
    cv::KeyPoint::convert(matched_kp_1_kp_2.second, pts2);

    // Bearing vectors
    opengv::bearingVectors_t bv1, bv2;
    calibrateKeypoints(pts1, pts2, bv1, bv2);
    Adapter adapter_mono(bv1, bv2);

    geometry_msgs::msg::PoseStamped pose_estimation;
    pose_estimation.pose.orientation.w = 1.0;
    geometry_msgs::msg::Pose relative_pose_estimate = pose_estimation.pose;

    switch (pose_estimator_) {
      case 0: { // 5-point (Nister) WITH RANSAC
        static constexpr size_t MIN = 5;
        if (adapter_mono.getNumberCorrespondences() >= MIN) {
          std::shared_ptr<RansacProblem> problem_ptr(
              new RansacProblem(adapter_mono, RansacProblem::NISTER));
          opengv::sac::Ransac<RansacProblem> ransac;
          ransac.sac_model_ = problem_ptr;
          ransac.threshold_ = 1e-3;
          ransac.max_iterations_ = 1000;
          if (ransac.computeModel()) {
            geometry_msgs::msg::Pose rel = eigen2Pose(ransac.model_coefficients_);
            relative_pose_estimate = rel;
          } else {
            RCLCPP_WARN(get_logger(), "5pt RANSAC failed.");
          }
        } else {
          RCLCPP_WARN(get_logger(), "Not enough correspondences for 5pt.");
        }
        break;
      }
      case 1: { // 8-point (Longuet–Higgins) WITH RANSAC
        static constexpr size_t MIN = 8;
        if (adapter_mono.getNumberCorrespondences() >= MIN) {
          std::shared_ptr<RansacProblem> problem_ptr(
              new RansacProblem(adapter_mono, RansacProblem::EIGHTPT));
          opengv::sac::Ransac<RansacProblem> ransac;
          ransac.sac_model_ = problem_ptr;
          ransac.threshold_ = 1e-3;
          ransac.max_iterations_ = 1000;
          if (ransac.computeModel()) {
            geometry_msgs::msg::Pose rel = eigen2Pose(ransac.model_coefficients_);
            relative_pose_estimate = rel;
          } else {
            RCLCPP_WARN(get_logger(), "8pt RANSAC failed.");
          }
        } else {
          RCLCPP_WARN(get_logger(), "Not enough correspondences for 8pt.");
        }
        break;
      }
      case 2: { // 2-point (known R) WITH RANSAC (translation-only)
        static constexpr size_t MIN = 2;
        if (adapter_mono.getNumberCorrespondences() >= MIN) {
          // Known rotation from GT between frames
          tf2::Transform Tcurr, Tprev;
          tf2::convert(curr_pose_.pose, Tcurr);
          tf2::convert(prev_pose_.pose, Tprev);
          geometry_msgs::msg::Transform tf = tf2::toMsg(Tprev.inverseTimes(Tcurr));
          Eigen::Matrix3d Rgt = tf2::transformToEigen(tf).rotation();
          adapter_mono.setR12(Rgt);

          // RANSAC translation-only
          std::shared_ptr<RansacProblemGivenRot> problem_ptr(
              new RansacProblemGivenRot(adapter_mono));
          opengv::sac::Ransac<RansacProblemGivenRot> ransac;
          ransac.sac_model_ = problem_ptr;
          ransac.threshold_ = 1e-3;
          ransac.max_iterations_ = 1000;
          if (ransac.computeModel()) {
            opengv::transformation_t T = opengv::transformation_t::Zero(3,4);
            T.block<3,3>(0,0) = Rgt;
            T.col(3) = ransac.model_coefficients_.col(3);
            relative_pose_estimate = eigen2Pose(T);
          } else {
            RCLCPP_WARN(get_logger(), "2pt RANSAC failed.");
          }
        } else {
          RCLCPP_WARN(get_logger(), "Not enough correspondences for 2pt.");
        }
        break;
      }
      case 3: {
        // ------------------------------------------------------------------
        // 3-point Arun (3D-3D) with depth + (optional) RANSAC
        // ------------------------------------------------------------------
        opengv::points_t cloud_1, cloud_2;
        cloud_1.reserve(bv1.size());
        cloud_2.reserve(bv2.size());

        auto in_bounds = [](int x, int y, const cv::Mat& m) {
          return (x >= 0 && y >= 0 && x < m.cols && y < m.rows);
        };

        const int N = static_cast<int>(std::min(bv1.size(), bv2.size()));
        for (int i = 0; i < N; ++i) {
          int u1 = static_cast<int>(std::floor(pts1[i].x));
          int v1 = static_cast<int>(std::floor(pts1[i].y));
          int u2 = static_cast<int>(std::floor(pts2[i].x));
          int v2 = static_cast<int>(std::floor(pts2[i].y));
          if (!in_bounds(u1, v1, prev_depth) || !in_bounds(u2, v2, depth)) continue;

          float d1 = prev_depth.at<float>(v1, u1);
          float d2 = depth.at<float>(v2, u2);
          if (!std::isfinite(d1) || !std::isfinite(d2) || d1 <= 0.f || d2 <= 0.f) continue;

          opengv::point_t p1 = bv1[i];
          opengv::point_t p2 = bv2[i];
          if (std::abs(p1(2)) < 1e-12 || std::abs(p2(2)) < 1e-12) continue;

          p1 /= p1(2);  p1 *= d1;   // prev 3D
          p2 /= p2(2);  p2 *= d2;   // curr 3D

          cloud_1.push_back(p1);
          cloud_2.push_back(p2);
        }

        if (cloud_1.size() < 3 || cloud_2.size() < 3) {
          RCLCPP_WARN(get_logger(),
                      "3pt Arun: not enough valid 3D correspondences (have %zu).",
                      cloud_1.size());
          break;
        }

        Adapter3D adapter_3d(cloud_1, cloud_2);

        if (!use_ransac_) {
          // ==== No RANSAC: Kabsch / Arun SVD ====
          Eigen::Vector3d mu1 = Eigen::Vector3d::Zero();
          Eigen::Vector3d mu2 = Eigen::Vector3d::Zero();
          for (size_t i = 0; i < cloud_1.size(); ++i) {
            mu1 += cloud_1[i];
            mu2 += cloud_2[i];
          }
          mu1 /= static_cast<double>(cloud_1.size());
          mu2 /= static_cast<double>(cloud_2.size());

          Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
          for (size_t i = 0; i < cloud_1.size(); ++i) {
            Eigen::Vector3d p = cloud_1[i] - mu1;
            Eigen::Vector3d q = cloud_2[i] - mu2;
            H += p * q.transpose(); // 3x3
          }

          Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
          Eigen::Matrix3d U = svd.matrixU();
          Eigen::Matrix3d V = svd.matrixV();

          Eigen::Matrix3d R = V * U.transpose();
          if (R.determinant() < 0) {
            // reflection fix
            V.col(2) *= -1;
            R = V * U.transpose();
          }
          Eigen::Vector3d t = mu2 - R * mu1;

          opengv::transformation_t T = opengv::transformation_t::Zero(3,4);
          T.block<3,3>(0,0) = R;
          T.col(3) = t;
          relative_pose_estimate = eigen2Pose(T);
        } else {
          // ==== With RANSAC: manual Kabsch-RANSAC on 3D-3D correspondences ====
          const double thresh = 0.03;   // 3 cm inlier threshold
          const int    iters  = 300;    // number of RANSAC iterations

          auto solve_kabsch = [](const std::vector<Eigen::Vector3d>& A,
                                 const std::vector<Eigen::Vector3d>& B)
                                 -> std::pair<Eigen::Matrix3d, Eigen::Vector3d> {
            const size_t n = A.size();
            Eigen::Vector3d muA = Eigen::Vector3d::Zero();
            Eigen::Vector3d muB = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < n; ++i) { muA += A[i]; muB += B[i]; }
            muA /= double(n); muB /= double(n);

            Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < n; ++i) {
              H += (A[i] - muA) * (B[i] - muB).transpose();
            }
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Matrix3d R = V * U.transpose();
            if (R.determinant() < 0) { Eigen::Matrix3d V2 = V; V2.col(2) *= -1; R = V2 * U.transpose(); }
            Eigen::Vector3d t = muB - R * muA;
            return {R, t};
          };

          auto residuals = [](const std::vector<Eigen::Vector3d>& A,
                              const std::vector<Eigen::Vector3d>& B,
                              const Eigen::Matrix3d& R,
                              const Eigen::Vector3d& t,
                              std::vector<double>& out) {
            const size_t n = A.size();
            out.resize(n);
            for (size_t i = 0; i < n; ++i) {
              out[i] = (R * A[i] + t - B[i]).norm();
            }
          };

          std::vector<Eigen::Vector3d> A, B;
          A.reserve(cloud_1.size()); B.reserve(cloud_2.size());
          for (size_t i = 0; i < cloud_1.size(); ++i) { A.push_back(cloud_1[i]); B.push_back(cloud_2[i]); }

          std::mt19937 rng(42);
          std::uniform_int_distribution<int> uni(0, static_cast<int>(A.size()) - 1);

          size_t best_inliers = 0;
          Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
          Eigen::Vector3d best_t = Eigen::Vector3d::Zero();

          for (int it = 0; it < iters; ++it) {
            int i1 = uni(rng), i2 = uni(rng), i3 = uni(rng);
            for (int tries = 0; tries < 10 && (i2 == i1); ++tries) i2 = uni(rng);
            for (int tries = 0; tries < 10 && (i3 == i1 || i3 == i2); ++tries) i3 = uni(rng);
            if (i1 == i2 || i1 == i3 || i2 == i3) continue;

            std::vector<Eigen::Vector3d> Amin = {A[i1], A[i2], A[i3]};
            std::vector<Eigen::Vector3d> Bmin = {B[i1], B[i2], B[i3]};

            auto [Rmin, tmin] = solve_kabsch(Amin, Bmin);

            std::vector<double> res;
            residuals(A, B, Rmin, tmin, res);
            size_t inl = 0;
            for (double r : res) if (r < thresh) ++inl;

            if (inl > best_inliers) {
              best_inliers = inl;
              best_R = Rmin; best_t = tmin;
            }
          }

          if (best_inliers >= 3) {
            std::vector<double> res;
            residuals(A, B, best_R, best_t, res);
            std::vector<Eigen::Vector3d> Ainl, Binl;
            Ainl.reserve(best_inliers); Binl.reserve(best_inliers);
            for (size_t i = 0; i < A.size(); ++i) if (res[i] <  thresh) { Ainl.push_back(A[i]); Binl.push_back(B[i]); }
            if (Ainl.size() >= 3) {
              auto [Rref, tref] = solve_kabsch(Ainl, Binl);
              best_R = Rref; best_t = tref;
            }

            opengv::transformation_t T = opengv::transformation_t::Zero(3,4);
            T.block<3,3>(0,0) = best_R;
            T.col(3) = best_t;
            relative_pose_estimate = eigen2Pose(T);
          } else {
            RCLCPP_WARN(get_logger(), "3pt Arun RANSAC: not enough inliers.");
          }

        }
        break;
      }
      default:
        RCLCPP_ERROR(get_logger(), "Wrong pose_estimator flag!");
    }

    // scale translation for 2D–2D only (visualization)
    if (pose_estimator_ < 3) {
      this->scaleTranslation(relative_pose_estimate.position, prev_pose_, curr_pose_);
    }

    // Publish absolute estimate at current frame
    this->updatePoseEstimate(prev_pose_.pose, relative_pose_estimate, pose_estimation.pose);
    pose_estimation.header.frame_id = "world";
    pub_pose_estimation_->publish(pose_estimation);

    // Evaluate RPE
    tf2::Transform gt_prev, gt_curr, est_prev, est_curr;
    tf2::convert(pose_estimation.pose, est_curr);
    tf2::convert(curr_pose_.pose, gt_curr);
    tf2::convert(prev_pose_.pose, est_prev);
    tf2::convert(prev_pose_.pose, gt_prev);
    this->evaluateRPE(gt_prev, gt_curr, est_prev, est_curr);

    // Save for next iter
    static cv::Mat prev_bgr_local; // just to keep structure identical
    prev_bgr_local = bgr.clone();
    prev_bgr = bgr.clone();
    prev_depth = depth.clone();
    prev_pose_ = curr_pose_;
  }

  // ===================== NEW: RGB-only callback =====================
  void rgbOnlyCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg) {
    cv::Mat bgr;
    try {
      bgr = cv_bridge::toCvShare(rgb_msg, "bgr8")->image;
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "Could not convert rgb image.");
      return;
    }

    static cv::Mat prev_bgr = bgr.clone();

    // Track features
    std::pair<std::vector<cv::KeyPoint>, std::vector<cv::KeyPoint>> matched_kp_1_kp_2;
    feature_tracker_->trackFeatures(prev_bgr, bgr, &matched_kp_1_kp_2, false, show_images_);
    std::vector<cv::Point2f> pts1, pts2;
    cv::KeyPoint::convert(matched_kp_1_kp_2.first,  pts1);
    cv::KeyPoint::convert(matched_kp_1_kp_2.second, pts2);

    // Bearing vectors
    opengv::bearingVectors_t bv1, bv2;
    calibrateKeypoints(pts1, pts2, bv1, bv2);
    Adapter adapter_mono(bv1, bv2);

    geometry_msgs::msg::PoseStamped pose_estimation;
    pose_estimation.pose.orientation.w = 1.0;
    geometry_msgs::msg::Pose relative_pose_estimate = pose_estimation.pose;

    switch (pose_estimator_) {
      case 0: { // 5-point
        static constexpr size_t MIN = 5;
        if (adapter_mono.getNumberCorrespondences() >= MIN) {
          std::shared_ptr<RansacProblem> problem_ptr(
              new RansacProblem(adapter_mono, RansacProblem::NISTER));
          opengv::sac::Ransac<RansacProblem> ransac;
          ransac.sac_model_ = problem_ptr;
          ransac.threshold_ = 1e-3;
          ransac.max_iterations_ = 1000;
          if (ransac.computeModel()) {
            geometry_msgs::msg::Pose rel = eigen2Pose(ransac.model_coefficients_);
            relative_pose_estimate = rel;
          } else {
            RCLCPP_WARN(get_logger(), "5pt RANSAC failed (RGB-only).");
          }
        } else {
          RCLCPP_WARN(get_logger(), "Not enough correspondences for 5pt (RGB-only).");
        }
        break;
      }
      case 1: { // 8-point
        static constexpr size_t MIN = 8;
        if (adapter_mono.getNumberCorrespondences() >= MIN) {
          std::shared_ptr<RansacProblem> problem_ptr(
              new RansacProblem(adapter_mono, RansacProblem::EIGHTPT));
          opengv::sac::Ransac<RansacProblem> ransac;
          ransac.sac_model_ = problem_ptr;
          ransac.threshold_ = 1e-3;
          ransac.max_iterations_ = 1000;
          if (ransac.computeModel()) {
            geometry_msgs::msg::Pose rel = eigen2Pose(ransac.model_coefficients_);
            relative_pose_estimate = rel;
          } else {
            RCLCPP_WARN(get_logger(), "8pt RANSAC failed (RGB-only).");
          }
        } else {
          RCLCPP_WARN(get_logger(), "Not enough correspondences for 8pt (RGB-only).");
        }
        break;
      }
      case 2: { // 2-point known R
        RCLCPP_WARN(get_logger(), "2pt (given R) requires GT rotation; skipping in RGB-only.");
        break;
      }
      case 3: {
        RCLCPP_WARN(get_logger(), "3pt Arun needs depth; skipping in RGB-only.");
        break;
      }
      default:
        RCLCPP_ERROR(get_logger(), "Wrong pose_estimator flag!");
    }

    // Scale translation for 2D–2D only
    if (pose_estimator_ < 3) {
      this->scaleTranslation(relative_pose_estimate.position, prev_pose_, curr_pose_);
    }

    // Publish absolute estimate at current frame
    this->updatePoseEstimate(prev_pose_.pose, relative_pose_estimate, pose_estimation.pose);
    pose_estimation.header.frame_id = "world";
    pub_pose_estimation_->publish(pose_estimation);

    // Evaluate RPE (will be NaN without GT; ok)
    tf2::Transform gt_prev, gt_curr, est_prev, est_curr;
    tf2::convert(pose_estimation.pose, est_curr);
    tf2::convert(prev_pose_.pose, est_prev);
    this->evaluateRPE(gt_prev, gt_curr, est_prev, est_curr);

    // Save for next iter (chain our own estimate if no GT arrives)
    prev_bgr = bgr.clone();
    prev_pose_ = pose_estimation;
  }
  // =================== end: RGB-only callback =======================
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  FLAGS_minloglevel = 0;
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  auto node = std::make_shared<PoseEstimator>();
  node->run();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  rclcpp::Rate r(100);
  while (rclcpp::ok()) {
    executor.spin_once();
    cv::waitKey(1);
    r.sleep();
  }
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
