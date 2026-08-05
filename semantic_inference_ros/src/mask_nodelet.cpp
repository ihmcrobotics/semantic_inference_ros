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

/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */

// Copyright (c) 2026, IHMC Robotics Lab.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rmw/qos_profiles.h>

#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace semantic_inference
{

class MaskNode : public rclcpp::Node
{
public:
  explicit MaskNode(const rclcpp::NodeOptions& options)
      : rclcpp::Node("mask", options)
  {
    declare_parameter<std::string>("mask_path", "");

    const std::string mask_path =
        get_parameter("mask_path").as_string();

    if (mask_path.empty())
    {
      RCLCPP_FATAL(
          get_logger(),
          "Parameter 'mask_path' is required");

      throw std::runtime_error(
          "mask_path parameter was not specified");
    }

    RCLCPP_INFO(
        get_logger(),
        "Reading mask from %s",
        mask_path.c_str());

    mask_ = cv::imread(
        mask_path,
        cv::IMREAD_GRAYSCALE);

    if (mask_.empty())
    {
      RCLCPP_FATAL(
          get_logger(),
          "Failed to load mask from '%s'",
          mask_path.c_str());

      throw std::runtime_error(
          "Invalid mask: loaded image is empty");
    }

    sub_ = image_transport::create_subscription(
        this,
        "input/image_raw",
        std::bind(
            &MaskNode::callback,
            this,
            std::placeholders::_1),
        "raw",
        rmw_qos_profile_sensor_data);

    pub_ = image_transport::create_publisher(
        this,
        "masked/image_raw");

    RCLCPP_INFO(
        get_logger(),
        "Mask node initialized");
  }

private:
  void callback(
      const sensor_msgs::msg::Image::ConstSharedPtr& msg)
  {
    if (!msg)
    {
      RCLCPP_ERROR(
          get_logger(),
          "Received a null image");
      return;
    }

    cv_bridge::CvImageConstPtr image_ptr;

    try
    {
      image_ptr = cv_bridge::toCvShare(msg);
    }
    catch (const cv_bridge::Exception& exception)
    {
      RCLCPP_ERROR(
          get_logger(),
          "cv_bridge exception: %s",
          exception.what());
      return;
    }

    if (mask_.rows != image_ptr->image.rows ||
        mask_.cols != image_ptr->image.cols)
    {
      RCLCPP_ERROR(
          get_logger(),
          "Mask size %dx%d does not match image size %dx%d",
          mask_.cols,
          mask_.rows,
          image_ptr->image.cols,
          image_ptr->image.rows);
      return;
    }

    if (!result_image_ ||
        result_image_->image.rows != image_ptr->image.rows ||
        result_image_->image.cols != image_ptr->image.cols ||
        result_image_->image.type() != image_ptr->image.type())
    {
      result_image_ =
          std::make_shared<cv_bridge::CvImage>();

      result_image_->encoding =
          image_ptr->encoding;

      result_image_->image =
          cv::Mat(
              image_ptr->image.rows,
              image_ptr->image.cols,
              image_ptr->image.type());
    }

    result_image_->header =
        image_ptr->header;

    result_image_->image.setTo(0);

    cv::bitwise_and(
        image_ptr->image,
        image_ptr->image,
        result_image_->image,
        mask_);

    pub_.publish(
        result_image_->toImageMsg());
  }

  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;

  cv_bridge::CvImagePtr result_image_;
  cv::Mat mask_;
};

}  // namespace semantic_inference

RCLCPP_COMPONENTS_REGISTER_NODE(
    semantic_inference::MaskNode)