// Portions of the following code and their modifications are originally from
// https://github.com/MIT-SPARK/semantic_inference and are licensed under the
// following license:
/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

// Copyright (c) 2026, IHMC Robotics Lab.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "semantic_inference_ros/output_publisher.h"

#include <config_utilities/config.h>
#include <config_utilities/validation.h>

#include <semantic_inference/image_utilities.h>
#include <semantic_inference/logging.h>

#include <opencv2/imgproc.hpp>

#include <iomanip>
#include <memory>
#include <optional>

namespace semantic_inference
{

OutputPublisher::OutputPublisher(
    const Config& input_config,
    rclcpp::Node& node)
    : config_(config::checkValid(input_config)),
      image_recolor_(config_.recolor)
{
  if (config_.publish_labels)
  {
    label_pub_ = image_transport::create_publisher(
        &node,
        "semantic/image_raw");
  }

  if (config_.publish_color)
  {
    color_pub_ = image_transport::create_publisher(
        &node,
        "semantic_color/image_raw");
  }

  if (config_.publish_overlay)
  {
    overlay_pub_ = image_transport::create_publisher(
        &node,
        "semantic_overlay/image_raw");
  }

  if (config_.publish_panoptic)
  {
    panoptic_pub_ = image_transport::create_publisher(
        &node,
        "panoptic/image_raw");
  }

  if (!config_.open_vocab)
  {
    image_feature_pub_ =
        node.create_publisher<
            semantic_inference_msgs::msg::FeatureVectorStamped>(
            "image_feature",
            rclcpp::QoS(1));

    semantics_with_features_pub_ =
        node.create_publisher<
            semantic_inference_msgs::msg::FeatureImage>(
            "semantic_color/feature_image",
            rclcpp::QoS(1));
  }
}

void OutputPublisher::resizePanoptic(const cv::Mat& panoptic) const
{
  if (!panoptic_image_ || panoptic_image_->encoding != "16SC1")
  {
    return;
  }

  // OpenCV does not support resizing CV_32S images directly.
  cv::Mat resized_panoptic;
  panoptic.convertTo(resized_panoptic, CV_16S);

  if (panoptic.rows != panoptic_image_->image.rows ||
      panoptic.cols != panoptic_image_->image.cols)
  {
    // Interpolating class or instance labels does not make sense, so use
    // nearest-neighbor interpolation.
    cv::resize(
        resized_panoptic,
        resized_panoptic,
        cv::Size(
            panoptic_image_->image.cols,
            panoptic_image_->image.rows),
        0.0,
        0.0,
        cv::INTER_NEAREST);
  }

  panoptic_image_->image = resized_panoptic;
}

void OutputPublisher::publish(
    const std_msgs::msg::Header& header,
    const cv::Mat& labels,
    const cv::Mat& color,
    const std::optional<cv::Mat>& panoptic)
{
  if (labels.empty() || color.empty())
  {
    SLOG(ERROR)
        << "Invalid inputs: color="
        << std::boolalpha
        << !color.empty()
        << ", labels="
        << !labels.empty();
    return;
  }

  if (!label_image_)
  {
    label_image_ = std::make_shared<cv_bridge::CvImage>();

    // Use signed 16-bit labels to distinguish semantic labels from unsigned
    // depth images.
    label_image_->encoding = "16SC1";
    label_image_->image =
        cv::Mat(color.rows, color.cols, CV_16SC1);
  }

  label_image_->header = header;
  image_recolor_.relabelImage(labels, label_image_->image);

  if (config_.publish_labels)
  {
    label_pub_.publish(label_image_->toImageMsg());
  }

  if (!config_.publish_color &&
      !config_.publish_overlay &&
      !config_.publish_panoptic &&
      config_.open_vocab)
  {
    return;
  }

  if (!panoptic_image_)
  {
    panoptic_image_ = std::make_shared<cv_bridge::CvImage>();
    panoptic_image_->encoding = "16SC1";
    panoptic_image_->image =
        cv::Mat(color.rows, color.cols, CV_16SC1);
  }

  panoptic_image_->header = header;

  if (panoptic.has_value())
  {
    resizePanoptic(*panoptic);
  }

  if (config_.publish_panoptic && panoptic.has_value())
  {
    panoptic_pub_.publish(panoptic_image_->toImageMsg());
  }

  if (!config_.publish_color &&
      !config_.publish_overlay &&
      config_.open_vocab)
  {
    return;
  }

  if (!color_image_)
  {
    color_image_ = std::make_shared<cv_bridge::CvImage>();
    color_image_->encoding = "rgb8";
    color_image_->image =
        cv::Mat(color.rows, color.cols, CV_8UC3);
  }

  color_image_->header = header;
  image_recolor_.recolorImage(
      label_image_->image,
      color_image_->image);

  if (config_.publish_color)
  {
    color_pub_.publish(color_image_->toImageMsg());
  }

  if (!config_.publish_overlay && config_.open_vocab)
  {
    return;
  }

  if (!overlay_image_)
  {
    overlay_image_ = std::make_shared<cv_bridge::CvImage>();
    overlay_image_->encoding = "rgb8";
    overlay_image_->image =
        cv::Mat(color.rows, color.cols, CV_8UC3);
  }

  overlay_image_->header = header;

  cv::addWeighted(
      color_image_->image,
      config_.overlay_alpha,
      color,
      1.0 - config_.overlay_alpha,
      0.0,
      overlay_image_->image);

  if (config_.publish_overlay)
  {
    overlay_pub_.publish(overlay_image_->toImageMsg());
  }

  if (!config_.open_vocab)
  {
    // Dummy image feature vector.
    semantic_inference_msgs::msg::FeatureVectorStamped features;
    features.header = header;

    if (image_feature_pub_)
    {
      image_feature_pub_->publish(features);
    }

    // Dummy feature image with no feature vectors. The image is the same as
    // the semantic color image.
    semantic_inference_msgs::msg::FeatureImage feature_image;
    feature_image.header = header;
    feature_image.image = *color_image_->toImageMsg();

    if (semantics_with_features_pub_)
    {
      semantics_with_features_pub_->publish(feature_image);
    }
  }
}

void declare_config(OutputPublisher::Config& config)
{
  using namespace config;

  name("OutputPublisher::Config");

  field(config.recolor, "recolor");
  field(config.publish_labels, "publish_labels");
  field(config.publish_color, "publish_color");
  field(config.publish_overlay, "publish_overlay");
  field(config.publish_panoptic, "publish_panoptic");
  field(config.overlay_alpha, "overlay_alpha");
  field(config.open_vocab, "open_vocab");
}

}  // namespace semantic_inference