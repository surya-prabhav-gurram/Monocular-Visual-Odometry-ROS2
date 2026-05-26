cmake_minimum_required(VERSION 3.8)
project(lab_3)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(lk_feature_tracker_obj OBJECT
  src/lk_feature_tracker.cpp
)
target_include_directories(lk_feature_tracker_obj PRIVATE include)
target_link_libraries(lk_feature_tracker_obj ${OpenCV_LIBS} glog::glog)

add_executable(track_features
  src/track_features.cpp
  src/feature_tracker.cpp
  src/sift_feature_tracker.cpp
  src/orb_feature_tracker.cpp
  src/akaze_feature_tracker.cpp
  src/brisk_feature_tracker.cpp
  $<TARGET_OBJECTS:lk_feature_tracker_obj>
)
target_include_directories(track_features PRIVATE include)
ament_target_dependencies(track_features rclcpp sensor_msgs cv_bridge image_transport)
target_link_libraries(track_features ${OpenCV_LIBS} glog::glog)

add_executable(self_flow
  src/self_flow.cpp
  $<TARGET_OBJECTS:lk_feature_tracker_obj>
)
target_include_directories(self_flow PRIVATE include)
ament_target_dependencies(self_flow rclcpp sensor_msgs cv_bridge image_transport)
target_link_libraries(self_flow ${OpenCV_LIBS} glog::glog)

add_executable(lk_two_frames_demo
  src/lk_two_frames_demo.cpp
  $<TARGET_OBJECTS:lk_feature_tracker_obj>
)
target_include_directories(lk_two_frames_demo PRIVATE include)
ament_target_dependencies(lk_two_frames_demo rclcpp)
target_link_libraries(lk_two_frames_demo ${OpenCV_LIBS} glog::glog)

install(TARGETS
  track_features
  self_flow
  lk_two_frames_demo
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  images
  bags
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
