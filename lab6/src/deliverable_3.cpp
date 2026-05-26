#include <deque>
#include <optional>
#include <string>
#include <vector>
#include <utility>
#include <tuple>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "tf2/exceptions.h"

// YOLO result message
#include "ultralytics_ros/msg/yolo_result.hpp"

// GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/Marginals.h>

using gtsam::Symbol;
using gtsam::Pose3;
using gtsam::Point3;
using ultralytics_ros::msg::YoloResult;

class BearTriangulatorNode : public rclcpp::Node {
public:
  BearTriangulatorNode()
  : Node("deliverable_3"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_) {

    fixed_frame_  = this->declare_parameter<std::string>("fixed_frame",  "world");
    camera_frame_ = this->declare_parameter<std::string>("camera_frame", "openni_rgb_frame");

    caminfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/rgb/camera_info", 10,
      std::bind(&BearTriangulatorNode::caminfoCb, this, std::placeholders::_1));

    std::string det_topic =
      this->declare_parameter<std::string>("detections_topic", "/yolo_result");

    det_sub_ = create_subscription<YoloResult>(
      det_topic, 10,
      std::bind(&BearTriangulatorNode::detCb, this, std::placeholders::_1));

    auto qos_latched = rclcpp::QoS(1).reliable().transient_local();
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/cortex_marker_array", qos_latched);

    republish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(200),
      [this]() {
        if (!last_markers_.markers.empty()) {
          auto t = this->now();
          for (auto &m : last_markers_.markers) m.header.stamp = t;
          marker_pub_->publish(last_markers_);
        }
      });

    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("bear_pose", 1);
    cov_pub_  = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("bear_pose_cov", 1);
  }

private:
  struct Meas {
    double u{}, v{};
    Pose3  Twc;
  };

  void caminfoCb(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (msg->k.size() >= 9) {
      K_ = gtsam::Cal3_S2(msg->k[0], msg->k[4], 0.0, msg->k[2], msg->k[5]);
      have_caminfo_ = true;
    }
  }

  void detCb(const YoloResult::SharedPtr msg) {
    if (!have_caminfo_) return;

    geometry_msgs::msg::TransformStamped T;
    try {
      T = tf_buffer_.lookupTransform(fixed_frame_, camera_frame_, tf2::TimePointZero);
    } catch (...) { return; }

    Pose3 Twc(
      gtsam::Rot3::Quaternion(
        T.transform.rotation.w,
        T.transform.rotation.x,
        T.transform.rotation.y,
        T.transform.rotation.z),
      gtsam::Point3(
        T.transform.translation.x,
        T.transform.translation.y,
        T.transform.translation.z));

    const auto & det_arr = msg->detections.detections;

    for (const auto& d : det_arr) {
      if (d.results.empty()) continue;
      const auto& hyp = d.results.front().hypothesis;

      if (hyp.class_id.find("teddy") == std::string::npos &&
          hyp.class_id.find("bear")  == std::string::npos) continue;

      const auto& bb = d.bbox;
      meas_.push_back({bb.center.position.x, bb.center.position.y, Twc});
    }

    if (meas_.size() >= min_obs_) solveAndPublish();
  }

  void solveAndPublish() {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values init;

    const Symbol L('L', 0);
    init.insert(L, Point3(0.0, 0.0, 1.0));

    auto measNoise = gtsam::noiseModel::Isotropic::Sigma(2, reproj_sigma_);
    auto Kptr = boost::make_shared<gtsam::Cal3_S2>(K_);

    size_t i = 0;
    for (const auto& m : meas_) {
      Symbol Xi('x', i++);
      init.insert(Xi, m.Twc);

      auto priorNoise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 1e-6,1e-6,1e-6,1e-6,1e-6,1e-6).finished());
      graph.add(gtsam::PriorFactor<Pose3>(Xi, m.Twc, priorNoise));

      graph.add(gtsam::GenericProjectionFactor<Pose3, Point3, gtsam::Cal3_S2>(
        gtsam::Point2(m.u, m.v), measNoise, Xi, L, Kptr));
    }

    gtsam::LevenbergMarquardtOptimizer opt(graph, init);
    auto result = opt.optimize();

    Point3 Lhat = result.at<Point3>(L);

    geometry_msgs::msg::PoseStamped ps;
    ps.header.frame_id = fixed_frame_;
    ps.header.stamp = now();
    ps.pose.position.x = Lhat.x();
    ps.pose.position.y = Lhat.y();
    ps.pose.position.z = Lhat.z();
    ps.pose.orientation.w = 1.0;

    pose_pub_->publish(ps);
  }

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
  rclcpp::Subscription<YoloResult>::SharedPtr det_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr cov_pub_;
  visualization_msgs::msg::MarkerArray last_markers_;
  rclcpp::TimerBase::SharedPtr republish_timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  bool have_caminfo_{false};
  gtsam::Cal3_S2 K_;

  std::vector<Meas> meas_;
  std::string fixed_frame_{"world"};
  std::string camera_frame_{"openni_rgb_frame"};
  const size_t min_obs_ = 10;
  const double reproj_sigma_ = 1.5;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BearTriangulatorNode>());
  rclcpp::shutdown();
  return 0;
}
