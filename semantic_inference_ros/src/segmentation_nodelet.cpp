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
#include <image_transport/image_transport.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rmw/qos_profiles.h>

#include <semantic_inference/image_rotator.h>
#include <semantic_inference/model_config.h>
#include <semantic_inference/segmenter.h>

#include <sensor_msgs/msg/image.hpp>

#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>

#include "semantic_inference_ros/output_publisher.h"
#include "semantic_inference_ros/ros_log_sink.h"
#include "semantic_inference_ros/worker.h"

namespace semantic_inference
{

class SegmentationNode : public rclcpp::Node
{
public:
  using ImageMessage = sensor_msgs::msg::Image;
  using ImageMessagePtr = ImageMessage::ConstSharedPtr;
  using ImageWorker = Worker<ImageMessagePtr>;

  struct Config
  {
    Segmenter::Config segmenter;
    OutputPublisher::Config output;
    WorkerConfig worker;
    ImageRotator::Config image_rotator;
  };

  explicit SegmentationNode(const rclcpp::NodeOptions& options);

  ~SegmentationNode() override;

private:
  void runSegmentation(const ImageMessagePtr& msg);

  Config config_;

  std::unique_ptr<Segmenter> segmenter_;
  ImageRotator image_rotator_;
  std::unique_ptr<ImageWorker> worker_;

  std::unique_ptr<OutputPublisher> output_pub_;
  image_transport::Subscriber sub_;
};

void declare_config(SegmentationNode::Config& config)
{
  using namespace config;

  name("SegmentationNode::Config");
  field(config.segmenter, "segmenter");
  field(config.output, "output");
  field(config.worker, "worker");
  field(config.image_rotator, "image_rotator");
}

SegmentationNode::SegmentationNode(
    const rclcpp::NodeOptions& options)
    : rclcpp::Node("segmentation", options)
{
  logging::Logger::addSink(
    "ros",
    std::make_shared<RosLogSink>(get_logger()));

  config_ = config::fromContext<Config>();

  SLOG(INFO) << "\n" << config::toString(config_);
  config::checkValid(config_);

  try
  {
    segmenter_ =
        std::make_unique<Segmenter>(
            config_.segmenter);
  }
  catch (const std::exception& exception)
  {
    SLOG(ERROR)
        << "Exception while creating segmenter: "
        << exception.what();

    throw;
  }

  image_rotator_ =
      ImageRotator(config_.image_rotator);

  output_pub_ =
      std::make_unique<OutputPublisher>(
          config_.output,
          *this);

  worker_ = std::make_unique<ImageWorker>(
      config_.worker,
      [this](const auto& msg)
      {
        runSegmentation(msg);
      },
      [](const auto& msg)
      {
        return rclcpp::Time(msg->header.stamp);
      });

  sub_ = image_transport::create_subscription(
      this,
      "color/image_raw",
      [this](const ImageMessagePtr& msg)
      {
        if (!msg)
        {
          RCLCPP_ERROR(
              get_logger(),
              "Received a null color image");
          return;
        }

        worker_->addMessage(msg);
      },
      "raw",
      rmw_qos_profile_sensor_data);

  RCLCPP_INFO(
      get_logger(),
      "Segmentation node initialized");
}

SegmentationNode::~SegmentationNode()
{
  if (worker_)
  {
    worker_->stop();
  }
}

void SegmentationNode::runSegmentation(
    const ImageMessagePtr& msg)
{
  if (!msg)
  {
    RCLCPP_ERROR(
        get_logger(),
        "Cannot run segmentation on a null image");
    return;
  }

  cv_bridge::CvImageConstPtr image_ptr;

  try
  {
    image_ptr =
        cv_bridge::toCvShare(
            msg,
            "rgb8");
  }
  catch (const cv_bridge::Exception& exception)
  {
    SLOG(ERROR)
        << "cv_bridge exception: "
        << exception.what();
    return;
  }

  SLOG(DEBUG)
      << "Encoding: "
      << image_ptr->encoding
      << " size: "
      << image_ptr->image.cols
      << " x "
      << image_ptr->image.rows
      << " x "
      << image_ptr->image.channels()
      << " is right type? "
      << (
          image_ptr->image.type() == CV_8UC3
              ? "yes"
              : "no");

  const auto rotated =
      image_rotator_.rotate(
          image_ptr->image);

  const auto result =
      segmenter_->infer(rotated);

  if (!result)
  {
    SLOG(ERROR)
        << "Failed to run semantic inference";
    return;
  }

  const auto derotated_labels =
      image_rotator_.rotate(
          result.labels);

  output_pub_->publish(
      image_ptr->header,
      derotated_labels,
      image_ptr->image,
      result.panoptic_ids);
}

}  // namespace semantic_inference

RCLCPP_COMPONENTS_REGISTER_NODE(
    semantic_inference::SegmentationNode)