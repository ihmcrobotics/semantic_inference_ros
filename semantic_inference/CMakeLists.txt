cmake_minimum_required(VERSION 3.16)
project(semantic_inference VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(SEMANTIC_INFERENCE_USE_TRT "Build with TensorRT" OFF)
option(SEMANTIC_INFERENCE_BUILD_TESTS "Build tests" ON)
option(BUILD_SHARED_LIBS "Build shared libs" ON)

# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------

find_package(ament_cmake REQUIRED)
find_package(config_utilities REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# ----------------------------------------------------------------------
# TensorRT manual discovery (Jetson: no TensorRTConfig.cmake available)
# ----------------------------------------------------------------------

if(SEMANTIC_INFERENCE_USE_TRT)
  find_path(
    TensorRT_INCLUDE_DIR
    NvInfer.h
    PATHS
      /usr/include
      /usr/include/aarch64-linux-gnu
  )

  find_library(
    TensorRT_LIBRARY
    nvinfer
    PATHS
      /usr/lib
      /usr/lib/aarch64-linux-gnu
  )

  find_library(
    TensorRT_ONNXPARSER_LIBRARY
    nvonnxparser
    PATHS
      /usr/lib
      /usr/lib/aarch64-linux-gnu
  )

  find_library(
    TensorRT_INFER_PLUGIN_LIBRARY
    nvinfer_plugin
    PATHS
      /usr/lib
      /usr/lib/aarch64-linux-gnu
  )

  if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    message(STATUS "Found TensorRT: ${TensorRT_LIBRARY}")
    set(ENABLE_TENSORRT_CXX_VALUE 1)

    # TensorRT core
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(
      TensorRT::TensorRT
      PROPERTIES
        IMPORTED_LOCATION ${TensorRT_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR}
    )

    # TensorRT plugin
    if(TensorRT_INFER_PLUGIN_LIBRARY)
      add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
      set_target_properties(
        TensorRT::nvinfer_plugin
        PROPERTIES
          IMPORTED_LOCATION ${TensorRT_INFER_PLUGIN_LIBRARY}
          INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR}
      )
    endif()

    # TensorRT ONNX parser
    if(TensorRT_ONNXPARSER_LIBRARY)
      add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
      set_target_properties(
        TensorRT::nvonnxparser
        PROPERTIES
          IMPORTED_LOCATION ${TensorRT_ONNXPARSER_LIBRARY}
          INTERFACE_INCLUDE_DIRECTORIES ${TensorRT_INCLUDE_DIR}
      )
    endif()
  else()
    message(WARNING "TensorRT headers or libs not found. Inference disabled.")
    set(ENABLE_TENSORRT_CXX_VALUE 0)
  endif()
else()
  message(STATUS "Disabling TensorRT! Inference will not work!")
  set(ENABLE_TENSORRT_CXX_VALUE 0)
endif()

# ----------------------------------------------------------------------
# Generated configuration
# ----------------------------------------------------------------------

configure_file(
  cmake/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/semantic_inference_config.h
)

# ----------------------------------------------------------------------
# Library
# ----------------------------------------------------------------------

add_library(
  ${PROJECT_NAME}
  src/image_recolor.cpp
  src/image_rotator.cpp
  src/image_utilities.cpp
  src/logging.cpp
  src/model_config.cpp
  src/segmenter.cpp
  src/panoptic_extractor.cpp
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    config_utilities::config_utilities
    ${OpenCV_LIBRARIES}
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

set_property(
  TARGET ${PROJECT_NAME}
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

add_library(
  semantic_inference::${PROJECT_NAME}
  ALIAS ${PROJECT_NAME}
)

# ----------------------------------------------------------------------
# Optional TensorRT implementation
# ----------------------------------------------------------------------

if(ENABLE_TENSORRT_CXX_VALUE)
  target_sources(
    ${PROJECT_NAME}
    PRIVATE
      src/trt_utilities.cpp
      src/model.cpp
  )

  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      ${CUDA_INCLUDE_DIRS}
  )

  target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
      TensorRT::TensorRT
      ${CUDA_LIBRARIES}
  )

  if(TARGET TensorRT::nvinfer_plugin)
    target_link_libraries(
      ${PROJECT_NAME}
      PRIVATE
        TensorRT::nvinfer_plugin
    )
  endif()

  if(TARGET TensorRT::nvonnxparser)
    target_link_libraries(
      ${PROJECT_NAME}
      PRIVATE
        TensorRT::nvonnxparser
    )
  endif()
endif()

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

if(SEMANTIC_INFERENCE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ----------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------

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
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# ----------------------------------------------------------------------
# Ament exports
# ----------------------------------------------------------------------

ament_export_targets(
  export_${PROJECT_NAME}
  HAS_LIBRARY_TARGET
)

ament_export_include_directories(include)

ament_export_dependencies(
  config_utilities
  OpenCV
)

ament_package()