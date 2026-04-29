#include "pose/MoveNetPoseEstimator.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

constexpr std::array<const char*, 17> kMoveNetKeypointNames = {
    "nose",         "left_eye",     "right_eye",   "left_ear",   "right_ear", "left_shoulder",
    "right_shoulder","left_elbow",  "right_elbow", "left_wrist", "right_wrist","left_hip",
    "right_hip",    "left_knee",    "right_knee",  "left_ankle", "right_ankle"};

float clamp01(float value) {
  return std::clamp(value, 0.0F, 1.0F);
}

}  // namespace

MoveNetPoseEstimator::MoveNetPoseEstimator(std::string model_path,
                                           int input_width,
                                           int input_height,
                                           float keypoint_score_threshold)
    : model_path_(std::move(model_path)),
      input_width_(input_width > 0 ? input_width : 192),
      input_height_(input_height > 0 ? input_height : 192),
      keypoint_score_threshold_(keypoint_score_threshold) {}

bool MoveNetPoseEstimator::init() {
  model_ready_ = false;
  last_output_shape_ = "unknown";
  if (model_path_.empty()) {
    status_message_ = "pose=model_path_empty";
    return true;
  }
  if (!std::filesystem::exists(model_path_)) {
    status_message_ = "pose=model_missing";
    return true;
  }

  const std::string extension = std::filesystem::path(model_path_).extension().string();
  if (extension != ".tflite" && extension != ".onnx") {
    status_message_ = "pose=unexpected_model_extension";
    return true;
  }

  try {
    if (extension == ".onnx") {
      net_ = cv::dnn::readNetFromONNX(model_path_);
    } else {
      net_ = cv::dnn::readNetFromTFLite(model_path_);
    }
    if (net_.empty()) {
      status_message_ = "pose=load_failed_empty_net";
      return true;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    model_ready_ = true;
    status_message_ = extension == ".onnx" ? "pose=model_loaded_onnx" : "pose=model_loaded_tflite";
    return true;
  } catch (const cv::Exception& e) {
    std::cerr << "[Pose] failed to load model: " << model_path_ << std::endl;
    std::cerr << "[Pose] OpenCV exception: " << e.what() << std::endl;
    status_message_ = "pose=load_failed";
    return true;
  }
}

PoseFrameResult MoveNetPoseEstimator::estimate(const cv::Mat& frame_bgr) {
  PoseFrameResult result;
  if (!model_ready_ || frame_bgr.empty()) {
    return result;
  }

  try {
    cv::Mat resized_bgr;
    cv::resize(frame_bgr, resized_bgr, cv::Size(input_width_, input_height_));

    cv::Mat resized_rgb;
    cv::cvtColor(resized_bgr, resized_rgb, cv::COLOR_BGR2RGB);

    cv::Mat input_blob =
        cv::dnn::blobFromImage(resized_rgb, 1.0, cv::Size(input_width_, input_height_),
                               cv::Scalar(), false, false, CV_32F);
    net_.setInput(input_blob);

    cv::Mat output = net_.forward();
    last_output_shape_ = shapeString(output);

    cv::Mat output_float;
    if (output.depth() == CV_32F) {
      output_float = output;
    } else {
      output.convertTo(output_float, CV_32F);
    }

    const std::size_t expected_values = kMoveNetKeypointNames.size() * 3;
    if (output_float.total() < expected_values) {
      status_message_ = "pose=unexpected_output shape=" + last_output_shape_;
      return result;
    }

    const float* data = output_float.ptr<float>();
    result.keypoints.reserve(kMoveNetKeypointNames.size());
    for (std::size_t index = 0; index < kMoveNetKeypointNames.size(); ++index) {
      const std::size_t offset = index * 3;
      const float y_norm = clamp01(data[offset + 0]);
      const float x_norm = clamp01(data[offset + 1]);
      const float score = data[offset + 2];

      PoseKeypoint keypoint;
      keypoint.name = kMoveNetKeypointNames[index];
      keypoint.score = score;
      keypoint.point =
          cv::Point2f(x_norm * static_cast<float>(frame_bgr.cols), y_norm * static_cast<float>(frame_bgr.rows));
      result.keypoints.push_back(std::move(keypoint));

      if (score >= keypoint_score_threshold_) {
        result.has_pose = true;
      }
    }

    status_message_ = "pose=model_ready shape=" + last_output_shape_;
    return result;
  } catch (const cv::Exception& e) {
    std::cerr << "[Pose] forward failed for model: " << model_path_ << std::endl;
    std::cerr << "[Pose] OpenCV exception: " << e.what() << std::endl;
    model_ready_ = false;
    status_message_ = "pose=forward_failed";
    return result;
  }
}

std::string MoveNetPoseEstimator::shapeString(const cv::Mat& tensor) {
  if (tensor.empty()) {
    return "empty";
  }
  std::ostringstream oss;
  for (int dim = 0; dim < tensor.dims; ++dim) {
    if (dim > 0) {
      oss << "x";
    }
    oss << tensor.size[dim];
  }
  return oss.str();
}

}  // namespace asdun
