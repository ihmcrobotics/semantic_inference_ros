/* -----------------------------------------------------------------------------
 * BSD 3-Clause License
 *
 * Copyright (c) 2021-2024, Massachusetts Institute of Technology.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

// Copyright (c) 2025, IHMC Robotics Lab.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <config_utilities/config_utilities.h>
#include <config_utilities/parsing/context.h>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/synchronizer.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rmw/qos_profiles.h>

#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>

#include <sensor_msgs/msg/image.hpp>

#include <functional>
#include <memory>

#include "semantic_inference_ros/output_publisher.h"

namespace semantic_inference
{

class RgbdSegmentationNode : public rclcpp::Node
{
public:
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<
          sensor_msgs::msg::Image,
          sensor_msgs::msg::Image>;

  struct Config
  {
    Segmenter::Config segmenter;
    OutputPublisher::Config output;
    double depth_scale = 1.0;
  };

  explicit RgbdSegmentationNode(
      const rclcpp::NodeOptions& options);

private:
  void callback(
      const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
      const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg);

  Config config_;

  std::unique_ptr<Segmenter> segmenter_;

  image_transport::SubscriberFilter image_sub_;
  image_transport::SubscriberFilter depth_sub_;

  std::unique_ptr<
      message_filters::Synchronizer<SyncPolicy>>
      sync_;

  std::unique_ptr<OutputPublisher> output_pub_;
};

void declare_config(RgbdSegmentationNode::Config& config)
{
  using namespace config;

  name("RgbdSegmentationNode::Config");
  field(config.segmenter, "segmenter");
  field(config.output, "output");
  field(config.depth_scale, "depth_scale");

  check(
      config.depth_scale,
      GT,
      0.0,
      "depth_scale");
}

RgbdSegmentationNode::RgbdSegmentationNode(
    const rclcpp::NodeOptions& options)
    : rclcpp::Node("rgbd_segmentation", options)
{
  config_ = config::fromContext<Config>();

  RCLCPP_INFO(
      get_logger(),
      "%s",
      config::toString(config_).c_str());

  config::checkValid(config_);

  segmenter_ =
      std::make_unique<Segmenter>(
          config_.segmenter);

  output_pub_ =
      std::make_unique<OutputPublisher>(
          config_.output,
          *this);

  image_sub_.subscribe(
      this,
      "color/image_raw",
      "raw",
      rmw_qos_profile_sensor_data);

  depth_sub_.subscribe(
      this,
      "depth/image_rect",
      "raw",
      rmw_qos_profile_sensor_data);

  sync_ =
      std::make_unique<
          message_filters::Synchronizer<SyncPolicy>>(
          SyncPolicy(10),
          image_sub_,
          depth_sub_);

  sync_->registerCallback(
      std::bind(
          &RgbdSegmentationNode::callback,
          this,
          std::placeholders::_1,
          std::placeholders::_2));

  RCLCPP_INFO(
      get_logger(),
      "RGB-D segmentation node initialized");
}

void RgbdSegmentationNode::callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
{
  if (!rgb_msg || !depth_msg)
  {
    RCLCPP_ERROR(
        get_logger(),
        "Received a null RGB or depth image");
    return;
  }

  cv_bridge::CvImageConstPtr image_ptr;

  try
  {
    image_ptr = cv_bridge::toCvShare(
        rgb_msg,
        "rgb8");
  }
  catch (const cv_bridge::Exception& exception)
  {
    RCLCPP_ERROR(
        get_logger(),
        "cv_bridge exception while converting RGB image: %s",
        exception.what());
    return;
  }

  cv_bridge::CvImageConstPtr depth_image_ptr;

  try
  {
    depth_image_ptr =
        cv_bridge::toCvShare(depth_msg);
  }
  catch (const cv_bridge::Exception& exception)
  {
    RCLCPP_ERROR(
        get_logger(),
        "cv_bridge exception while converting depth image: %s",
        exception.what());
    return;
  }

  cv::Mat depth_image_float;

  if (depth_image_ptr->image.type() == CV_32FC1)
  {
    depth_image_float =
        depth_image_ptr->image;
  }
  else
  {
    depth_image_ptr->image.convertTo(
        depth_image_float,
        CV_32FC1,
        config_.depth_scale);
  }

  const auto result =
      segmenter_->infer(
          image_ptr->image,
          depth_image_float);

  if (!result)
  {
    RCLCPP_ERROR(
        get_logger(),
        "Failed to run RGB-D semantic inference");
    return;
  }

  output_pub_->publish(
      image_ptr->header,
      result.labels,
      image_ptr->image,
      result.panoptic_ids);
}

}  // namespace semantic_inference

RCLCPP_COMPONENTS_REGISTER_NODE(
    semantic_inference::RgbdSegmentationNode)