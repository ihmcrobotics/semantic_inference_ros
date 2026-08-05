cmake_minimum_required(VERSION 3.16)
project(semantic_inference_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

find_package(config_utilities REQUIRED)
find_package(semantic_inference REQUIRED)
find_package(semantic_inference_msgs REQUIRED)

ament_python_install_package(
  ${PROJECT_NAME}
  PACKAGE_DIR src/${PROJECT_NAME}
)

add_library(
  ${PROJECT_NAME} SHARED
  src/mask_nodelet.cpp
  src/rgbd_segmentation_nodelet.cpp
  src/segmentation_nodelet.cpp
  src/output_publisher.cpp
  src/recolor_nodelet.cpp
  src/republish_nodelet.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  rclcpp_components
  cv_bridge
  image_transport
  message_filters
  sensor_msgs
  std_msgs
  config_utilities
  semantic_inference_msgs
)

target_link_libraries(
  ${PROJECT_NAME}
  semantic_inference::semantic_inference
)

rclcpp_components_register_nodes(
  ${PROJECT_NAME}
  "semantic_inference::MaskNode"
  "semantic_inference::SegmentationNode"
  "semantic_inference::RgbdSegmentationNode"
  "semantic_inference::RecolorNode"
  "semantic_inference::RepublishNode"
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  PROGRAMS
    app/openset_segmentation_node
    app/clip_features_node
    app/clip_publisher_node
    app/language_embedding_node
    app/encode_vlm_labels_node
    app/vlm_features_node
    app/dummy_vlm_node
    app/navigation_prompt_service
    app/vlm_for_navigation_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(
  export_${PROJECT_NAME}
  HAS_LIBRARY_TARGET
)

ament_export_include_directories(include)

ament_export_dependencies(
  rclcpp
  rclcpp_components
  cv_bridge
  image_transport
  message_filters
  sensor_msgs
  std_msgs
  config_utilities
  semantic_inference
  semantic_inference_msgs
)

ament_package()