cmake_minimum_required(VERSION 3.8)
project(lab_6)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(ultralytics_ros REQUIRED)

find_package(GTSAM REQUIRED)
find_package(glog REQUIRED)

add_executable(deliverable_3 src/deliverable_3.cpp)

ament_target_dependencies(deliverable_3
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
  vision_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  ultralytics_ros
)

target_link_libraries(deliverable_3
  gtsam
  glog::glog
)

target_include_directories(deliverable_3 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS deliverable_3
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
