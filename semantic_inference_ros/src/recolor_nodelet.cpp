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

// Copyright (c) 2026, IHMC Robotics Lab.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <config_utilities/parsing/context.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <message_filters/sync_policies/exact_time.hpp>
#include <message_filters/synchronizer.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rmw/qos_profiles.h>

#include <sensor_msgs/msg/image.hpp>

#include <functional>
#include <memory>
#include <utility>

#include "semantic_inference_ros/output_publisher.h"
#include "semantic_inference_ros/ros_log_sink.h"
#include "semantic_inference_ros/worker.h"

namespace semantic_inference
{

struct ColorLabelPacket
{
  sensor_msgs::msg::Image::ConstSharedPtr color;
  sensor_msgs::msg::Image::ConstSharedPtr labels;
};

class RecolorNode : public rclcpp::Node
{
public:
  using ImageWorker = Worker<ColorLabelPacket>;

  using SyncPolicy =
      message_filters::sync_policies::ExactTime<
          sensor_msgs::msg::Image,
          sensor_msgs::msg::Image>;

  struct Config
  {
    OutputPublisher::Config output;
    WorkerConfig worker;
  };

  explicit RecolorNode(const rclcpp::NodeOptions& options);

  ~RecolorNode() override;

private:
  void publish(const ColorLabelPacket& packet) const;

  void callback(
      const sensor_msgs::msg::Image::ConstSharedPtr& color,
      const sensor_msgs::msg::Image::ConstSharedPtr& labels);

  Config config_;

  std::unique_ptr<ImageWorker> worker_;

  image_transport::SubscriberFilter color_sub_;
  image_transport::SubscriberFilter label_sub_;

  std::unique_ptr<
      message_filters::Synchronizer<SyncPolicy>>
      sync_;

  std::unique_ptr<OutputPublisher> pub_;
};

void declare_config(RecolorNode::Config& config)
{
  using namespace config;

  name("RecolorNode::Config");
  field(config.output, "output");
  field(config.worker, "worker");
}

RecolorNode::RecolorNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("recolor", options)
{ 
  logging::Logger::addSink(
    "ros",
    std::make_shared<RosLogSink>(get_logger()));

  config_ = config::fromContext<Config>();

  SLOG(INFO) << "\n" << config::toString(config_);
  config::checkValid(config_);

  pub_ = std::make_unique<OutputPublisher>(
      config_.output,
      *this);

  worker_ = std::make_unique<ImageWorker>(
      config_.worker,
      [this](const auto& packet)
      {
        publish(packet);
      },
      [](const auto& packet)
      {
        return rclcpp::Time(packet.color->header.stamp);
      });

  color_sub_.subscribe(
      this,
      "color/image_raw",
      "raw",
      rmw_qos_profile_sensor_data);

  label_sub_.subscribe(
      this,
      "labels/image_raw",
      "raw",
      rmw_qos_profile_sensor_data);

  sync_ =
      std::make_unique<
          message_filters::Synchronizer<SyncPolicy>>(
          SyncPolicy(10),
          color_sub_,
          label_sub_);

  sync_->registerCallback(
      std::bind(
          &RecolorNode::callback,
          this,
          std::placeholders::_1,
          std::placeholders::_2));

  RCLCPP_INFO(
      get_logger(),
      "Recolor node initialized");
}

RecolorNode::~RecolorNode()
{
  if (worker_)
  {
    worker_->stop();
  }
}

void RecolorNode::publish(
    const ColorLabelPacket& packet) const
{
  cv_bridge::CvImageConstPtr color_ptr;

  try
  {
    color_ptr = cv_bridge::toCvShare(
        packet.color,
        "rgb8");
  }
  catch (const cv_bridge::Exception& exception)
  {
    RCLCPP_ERROR(
        get_logger(),
        "cv_bridge exception while converting color image: %s",
        exception.what());
    return;
  }

  cv_bridge::CvImageConstPtr label_ptr;

  try
  {
    label_ptr = cv_bridge::toCvShare(packet.labels);
  }
  catch (const cv_bridge::Exception& exception)
  {
    RCLCPP_ERROR(
        get_logger(),
        "cv_bridge exception while converting label image: %s",
        exception.what());
    return;
  }

  pub_->publish(
      color_ptr->header,
      label_ptr->image,
      color_ptr->image);
}

void RecolorNode::callback(
    const sensor_msgs::msg::Image::ConstSharedPtr& color,
    const sensor_msgs::msg::Image::ConstSharedPtr& labels)
{
  worker_->addMessage(
      ColorLabelPacket{
          color,
          labels});
}

}  // namespace semantic_inference

RCLCPP_COMPONENTS_REGISTER_NODE(
    semantic_inference::RecolorNode)