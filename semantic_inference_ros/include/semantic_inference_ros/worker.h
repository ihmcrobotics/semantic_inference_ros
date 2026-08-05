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
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
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

#pragma once

#include <config_utilities/config.h>
#include <rclcpp/rclcpp.hpp>
#include <semantic_inference/logging.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace semantic_inference
{

struct WorkerConfig
{
  std::size_t max_queue_size = 30;
  double image_separation_s = 0.0;
};

inline void declare_config(WorkerConfig& config)
{
  using namespace config;

  name("WorkerConfig");
  field(config.max_queue_size, "max_queue_size");
  field(config.image_separation_s, "image_separation_s");
}

template <typename T>
class Worker
{
public:
  using Callback = std::function<void(const T&)>;
  using StampGetter = std::function<rclcpp::Time(const T&)>;

  Worker(
      const WorkerConfig& config,
      Callback callback,
      StampGetter stamp_getter);

  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  Worker(Worker&&) = delete;
  Worker& operator=(Worker&&) = delete;

  void addMessage(const T& msg);

  void stop();

  const WorkerConfig config;

private:
  bool haveWork() const;

  void spin();

  const Callback callback_;
  const StampGetter stamp_getter_;

  mutable std::mutex mutex_;
  std::atomic<bool> should_shutdown_{false};
  std::unique_ptr<std::thread> spin_thread_;
  std::list<T> queue_;
};

template <typename T>
Worker<T>::Worker(
    const WorkerConfig& config,
    Callback callback,
    StampGetter stamp_getter)
    : config(config),
      callback_(std::move(callback)),
      stamp_getter_(std::move(stamp_getter)),
      spin_thread_(
          std::make_unique<std::thread>(
              &Worker<T>::spin,
              this))
{
}

template <typename T>
Worker<T>::~Worker()
{
  stop();
}

template <typename T>
void Worker<T>::stop()
{
  should_shutdown_.store(true);

  if (spin_thread_ && spin_thread_->joinable())
  {
    spin_thread_->join();
  }

  spin_thread_.reset();
}

template <typename T>
void Worker<T>::addMessage(const T& msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  while (
      config.max_queue_size > 0 &&
      queue_.size() >= config.max_queue_size)
  {
    queue_.pop_front();
  }

  queue_.push_back(msg);
}

template <typename T>
bool Worker<T>::haveWork() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return !queue_.empty();
}

template <typename T>
void Worker<T>::spin()
{
  using namespace std::chrono_literals;

  std::optional<rclcpp::Time> last_time;

  while (rclcpp::ok() && !should_shutdown_.load())
  {
    if (!haveWork())
    {
      std::this_thread::sleep_for(10ms);
      continue;
    }

    std::optional<T> message;

    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (queue_.empty())
      {
        continue;
      }

      const rclcpp::Time current_stamp =
          stamp_getter_(queue_.front());

      if (last_time.has_value())
      {
        const double current_difference_s =
            std::abs(
                (current_stamp - *last_time).seconds());

        SLOG(DEBUG)
            << "Current time difference: "
            << current_difference_s
            << " [s] (minimum: "
            << config.image_separation_s
            << " [s])";

        if (current_difference_s < config.image_separation_s)
        {
          queue_.pop_front();
          continue;
        }
      }

      last_time = current_stamp;
      message.emplace(std::move(queue_.front()));
      queue_.pop_front();
    }

    if (message.has_value())
    {
      callback_(*message);
    }
  }
}

}  // namespace semantic_inference