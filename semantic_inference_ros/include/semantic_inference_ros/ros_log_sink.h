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

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <semantic_inference/logging.h>

#include <sstream>
#include <utility>

namespace semantic_inference
{

struct RosLogSink : logging::LogSink
{
  explicit RosLogSink(
      rclcpp::Logger logger =
          rclcpp::get_logger("semantic_inference"))
      : logger_(std::move(logger))
  {
  }

  ~RosLogSink() override = default;

  void dispatch(
      const logging::LogEntry& entry) const override
  {
    std::stringstream stream;
    stream << entry.prefix() << entry.message();

    const std::string message = stream.str();

    switch (entry.level)
    {
      case logging::Level::WARNING:
        RCLCPP_WARN(
            logger_,
            "%s",
            message.c_str());
        break;

      case logging::Level::ERROR:
        RCLCPP_ERROR(
            logger_,
            "%s",
            message.c_str());
        break;

      case logging::Level::FATAL:
        RCLCPP_FATAL(
            logger_,
            "%s",
            message.c_str());
        break;

      case logging::Level::INFO:
        RCLCPP_INFO(
            logger_,
            "%s",
            message.c_str());
        break;

      case logging::Level::DEBUG:
      default:
        RCLCPP_DEBUG(
            logger_,
            "%s",
            message.c_str());
        break;
    }
  }

private:
  rclcpp::Logger logger_;
};

}  // namespace semantic_inference