// Copyright (c) 2026, IHMC Robotics Lab.
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

#include <semantic_inference/image_recolor.h>
#include <semantic_inference/panoptic_extractor.h>

#include <semantic_inference_msgs/msg/feature_image.hpp>
#include <semantic_inference_msgs/msg/feature_vector_stamped.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <functional>
#include <memory>

#include <opencv2/opencv.hpp>

namespace semantic_inference
{

class RepublishNode : public rclcpp::Node
{
public:
  struct Config
  {
    ImageRecolor::Config recolor;
    bool open_vocab = false;
  };

  explicit RepublishNode(const rclcpp::NodeOptions& options);

  ~RepublishNode() override = default;

private:
  void runGtRepublish(
      const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  Config config_;

  image_transport::Subscriber sub_;
  image_transport::Publisher panoptic_pub_;

  struct Publishers
  {
    rclcpp::Publisher<
        semantic_inference_msgs::msg::FeatureImage>::SharedPtr
        feature_image;

    rclcpp::Publisher<
        semantic_inference_msgs::msg::FeatureVectorStamped>::SharedPtr
        feature_vector;
  } pubs_;

  std::unique_ptr<PanopticExtractor> panoptic_extractor_;
  std::unique_ptr<ImageRecolor> image_recolor_;

  cv_bridge::CvImagePtr panoptic_image_;
  cv_bridge::CvImagePtr label_image_;
};

void declare_config(RepublishNode::Config& config)
{
  using namespace config;

  name("RepublishNode::Config");
  field(config.recolor, "recolor");
  field(config.open_vocab, "open_vocab");
}

RepublishNode::RepublishNode(
    const rclcpp::NodeOptions& options)
    : rclcpp::Node("republish", options)
{
  config_ = config::fromContext<Config>();
  config::checkValid(config_);

  image_recolor_ =
      std::make_unique<ImageRecolor>(config_.recolor);

  panoptic_extractor_ =
      std::make_unique<PanopticExtractor>();

  sub_ = image_transport::create_subscription(
      this,
      "semantic/image_raw",
      std::bind(
          &RepublishNode::runGtRepublish,
          this,
          std::placeholders::_1),
      "raw",
      rmw_qos_profile_sensor_data);

  RCLCPP_INFO(
      get_logger(),
      "Subscribed to semantic/image_raw");

  if (!config_.open_vocab)
  {
    pubs_.feature_image =
        create_publisher<
            semantic_inference_msgs::msg::FeatureImage>(
            "semantic_color/feature_image",
            rclcpp::QoS(1));

    pubs_.feature_vector =
        create_publisher<
            semantic_inference_msgs::msg::FeatureVectorStamped>(
            "image_feature",
            rclcpp::QoS(1));
  }

  panoptic_pub_ = image_transport::create_publisher(
      this,
      "panoptic/image_raw");

  RCLCPP_INFO(
      get_logger(),
      "Republish node initialized");
}

void RepublishNode::runGtRepublish(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  if (!msg)
  {
    RCLCPP_ERROR(
        get_logger(),
        "Received a null semantic image");
    return;
  }

  if (!panoptic_image_)
  {
    panoptic_image_ =
        std::make_shared<cv_bridge::CvImage>();

    panoptic_image_->encoding = "16SC1";
    panoptic_image_->image =
        cv::Mat(
            static_cast<int>(msg->height),
            static_cast<int>(msg->width),
            CV_16SC1);
  }

  if (!label_image_)
  {
    label_image_ =
        std::make_shared<cv_bridge::CvImage>();

    label_image_->encoding = "16SC1";
    label_image_->image =
        cv::Mat(
            static_cast<int>(msg->height),
            static_cast<int>(msg->width),
            CV_16SC1);
  }

  panoptic_image_->header = msg->header;
  label_image_->header = msg->header;

  cv_bridge::CvImageConstPtr input_image;

  try
  {
    input_image = cv_bridge::toCvShare(msg);
  }
  catch (const cv_bridge::Exception& exception)
  {
    RCLCPP_ERROR(
        get_logger(),
        "cv_bridge exception: %s",
        exception.what());
    return;
  }

  image_recolor_->labelImage(
      input_image->image,
      label_image_->image);

  cv::Mat label_image_reshaped;
  label_image_->image.convertTo(
      label_image_reshaped,
      CV_32S);

  panoptic_extractor_->extract(
      label_image_reshaped,
      panoptic_image_->image);

  panoptic_pub_.publish(
      panoptic_image_->toImageMsg());

  if (!config_.open_vocab)
  {
    semantic_inference_msgs::msg::FeatureImage feature_image;
    semantic_inference_msgs::msg::FeatureVectorStamped feature_vector;

    feature_image.header = msg->header;
    feature_image.image = *msg;

    feature_vector.header = msg->header;

    if (pubs_.feature_image)
    {
      pubs_.feature_image->publish(feature_image);
    }

    if (pubs_.feature_vector)
    {
      pubs_.feature_vector->publish(feature_vector);
    }
  }
}

}  // namespace semantic_inference

RCLCPP_COMPONENTS_REGISTER_NODE(
    semantic_inference::RepublishNode)