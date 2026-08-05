cmake_minimum_required(VERSION 3.16)
project(semantic_inference_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FeatureVector.msg"
  "msg/FeatureVectors.msg"
  "msg/FeatureVectorStamped.msg"
  "msg/FeatureVectorsStamped.msg"
  "msg/FeatureImage.msg"
  "msg/LevelFeatureVectorStamped.msg"
  "msg/NavigationPromptsEmbeddings.msg"
  "msg/ObjectsPromptPair.msg"
  "msg/VlmForNavigationDebug.msg"
  "msg/VlmForNavigationOutput.msg"
  "msg/VlmForNavigationRelationshipOutput.msg"

  "srv/EncodeFeature.srv"
  "srv/NavigationPrompt.srv"

  DEPENDENCIES
    sensor_msgs
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()