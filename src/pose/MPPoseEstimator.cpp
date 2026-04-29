#include "pose/MPPoseEstimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

constexpr std::array<const char*, 33> kMediaPipePoseNames = {
    "nose",         "left_eye_inner", "left_eye",       "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear",    "right_ear",      "mouth_left",     "mouth_right",     "left_shoulder",
    "right_shoulder",  "left_elbow",  "right_elbow",    "left_wrist",     "right_wrist",     "left_pinky",
    "right_pinky",     "left_index",  "right_index",    "left_thumb",     "right_thumb",     "left_hip",
    "right_hip",       "left_knee",   "right_knee",     "left_ankle",     "right_ankle",     "left_heel",
    "right_heel",      "left_foot_index", "right_foot_index"};

constexpr double kPi = 3.14159265358979323846;
constexpr float kPersonBoxPreEnlargeFactor = 1.0F;
constexpr float kPersonBoxEnlargeFactor = 1.25F;

float sigmoid(float value) {
  return 1.0F / (1.0F + std::exp(-value));
}

}  // namespace

MPPoseEstimator::MPPoseEstimator(std::string model_path, float score_threshold)
    : model_path_(std::move(model_path)), score_threshold_(score_threshold) {}

bool MPPoseEstimator::init() {
  model_ready_ = false;
  last_output_shapes_ = "unknown";
  if (model_path_.empty()) {
    status_message_ = "pose=model_path_empty";
    return true;
  }

  try {
    net_ = cv::dnn::readNet(model_path_);
    if (net_.empty()) {
      status_message_ = "pose=load_failed_empty_net";
      return true;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    model_ready_ = true;
    status_message_ = "pose=model_loaded";
    return true;
  } catch (const cv::Exception& e) {
    std::cerr << "[Pose/MPPose] failed to load model: " << model_path_ << std::endl;
    std::cerr << "[Pose/MPPose] OpenCV exception: " << e.what() << std::endl;
    status_message_ = "pose=load_failed";
    return true;
  }
}

PoseFrameResult MPPoseEstimator::estimate(const cv::Mat& frame_bgr, const MPPersonRegion& person_region) {
  PoseFrameResult result;
  if (!model_ready_ || frame_bgr.empty() || !person_region.valid()) {
    return result;
  }

  try {
    std::array<cv::Point2f, 4> person_keypoints = person_region.aux_keypoints;
    cv::Point2f mid_hip_point = person_keypoints[0];
    cv::Point2f full_body_point = person_keypoints[1];

    const float full_dist = std::max(static_cast<float>(cv::norm(mid_hip_point - full_body_point)), 1.0F);
    std::array<cv::Point2i, 2> full_bbox = {
        cv::Point(cvRound(mid_hip_point.x - full_dist), cvRound(mid_hip_point.y - full_dist)),
        cv::Point(cvRound(mid_hip_point.x + full_dist), cvRound(mid_hip_point.y + full_dist))};

    const cv::Point2f center_bbox((full_bbox[0].x + full_bbox[1].x) * 0.5F, (full_bbox[0].y + full_bbox[1].y) * 0.5F);
    const cv::Point2f wh_bbox(static_cast<float>(full_bbox[1].x - full_bbox[0].x),
                              static_cast<float>(full_bbox[1].y - full_bbox[0].y));
    const cv::Point2f new_half_size = wh_bbox * (kPersonBoxPreEnlargeFactor * 0.5F);
    full_bbox = {
        cv::Point(cvRound(center_bbox.x - new_half_size.x), cvRound(center_bbox.y - new_half_size.y)),
        cv::Point(cvRound(center_bbox.x + new_half_size.x), cvRound(center_bbox.y + new_half_size.y))};

    std::array<cv::Point2i, 2> person_bbox = full_bbox;
    person_bbox[0].x = std::clamp(person_bbox[0].x, 0, frame_bgr.cols);
    person_bbox[1].x = std::clamp(person_bbox[1].x, 0, frame_bgr.cols);
    person_bbox[0].y = std::clamp(person_bbox[0].y, 0, frame_bgr.rows);
    person_bbox[1].y = std::clamp(person_bbox[1].y, 0, frame_bgr.rows);

    const cv::Rect crop_rect(person_bbox[0].x, person_bbox[0].y, person_bbox[1].x - person_bbox[0].x,
                             person_bbox[1].y - person_bbox[0].y);
    if (crop_rect.width <= 0 || crop_rect.height <= 0) {
      status_message_ = "pose=invalid_roi";
      return result;
    }

    cv::Mat image = frame_bgr(crop_rect).clone();
    const int left = person_bbox[0].x - full_bbox[0].x;
    const int top = person_bbox[0].y - full_bbox[0].y;
    const int right = full_bbox[1].x - person_bbox[1].x;
    const int bottom = full_bbox[1].y - person_bbox[1].y;
    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    cv::Point2f pad_bias(static_cast<float>(person_bbox[0].x - left), static_cast<float>(person_bbox[0].y - top));
    mid_hip_point -= pad_bias;
    full_body_point -= pad_bias;

    const double radians =
        kPi / 2.0 - std::atan2(-(full_body_point.y - mid_hip_point.y), full_body_point.x - mid_hip_point.x);
    const double angle = radians * 180.0 / kPi;
    const cv::Mat rotation_matrix = cv::getRotationMatrix2D(mid_hip_point, angle, 1.0);

    cv::Mat rotated;
    cv::warpAffine(image, rotated, rotation_matrix, image.size());

    cv::Mat resized;
    cv::resize(rotated, resized, input_size_, 0.0, 0.0, cv::INTER_AREA);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    cv::Mat input_blob = makeNhwcBlob(resized);
    net_.setInput(input_blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.size() < 5) {
      status_message_ = "pose=unexpected_output_count";
      return result;
    }

    last_output_shapes_ = shapeString(outputs[0]) + "," + shapeString(outputs[1]) + "," + shapeString(outputs[2]) + "," +
                          shapeString(outputs[3]) + "," + shapeString(outputs[4]);

    cv::Mat conf_mat = outputs[1].reshape(1, 1);
    float conf = conf_mat.at<float>(0, 0);
    if (conf < score_threshold_) {
      status_message_ = "pose=low_confidence";
      return result;
    }

    cv::Mat landmarks = outputs[0].reshape(1, 39);
    if (landmarks.cols < 5) {
      status_message_ = "pose=unexpected_landmark_shape";
      return result;
    }

    const cv::Point2f roi_size(static_cast<float>(image.cols), static_cast<float>(image.rows));
    const cv::Point2f scale_factor(roi_size.x / static_cast<float>(input_size_.width),
                                   roi_size.y / static_cast<float>(input_size_.height));

    cv::Mat inverse_rotation_matrix;
    cv::invertAffineTransform(rotation_matrix, inverse_rotation_matrix);

    result.keypoints.reserve(33);
    for (int index = 0; index < 33; ++index) {
      const float* landmark_row = landmarks.ptr<float>(index);
      float x = (landmark_row[0] - input_size_.width * 0.5F) * scale_factor.x;
      float y = (landmark_row[1] - input_size_.height * 0.5F) * scale_factor.y;
      float z = landmark_row[2] * std::max(scale_factor.x, scale_factor.y);
      float visibility = sigmoid(landmark_row[3]);
      float presence = sigmoid(landmark_row[4]);
      float score = std::min(visibility, presence);

      const cv::Point2f rotated_point(mid_hip_point.x + x, mid_hip_point.y + y);
      cv::Point2f original_point(
          static_cast<float>(inverse_rotation_matrix.at<double>(0, 0) * rotated_point.x +
                             inverse_rotation_matrix.at<double>(0, 1) * rotated_point.y +
                             inverse_rotation_matrix.at<double>(0, 2)),
          static_cast<float>(inverse_rotation_matrix.at<double>(1, 0) * rotated_point.x +
                             inverse_rotation_matrix.at<double>(1, 1) * rotated_point.y +
                             inverse_rotation_matrix.at<double>(1, 2)));
      original_point += pad_bias;
      (void)z;

      PoseKeypoint keypoint;
      keypoint.name = kMediaPipePoseNames[static_cast<std::size_t>(index)];
      keypoint.point = original_point;
      keypoint.score = score;
      result.keypoints.push_back(std::move(keypoint));
      if (score >= score_threshold_) {
        result.has_pose = true;
      }
    }

    status_message_ = "pose=model_ready shape=" + last_output_shapes_;
    return result;
  } catch (const cv::Exception& e) {
    std::cerr << "[Pose/MPPose] forward failed for model: " << model_path_ << std::endl;
    std::cerr << "[Pose/MPPose] OpenCV exception: " << e.what() << std::endl;
    model_ready_ = false;
    status_message_ = "pose=forward_failed";
    return result;
  }
}

std::string MPPoseEstimator::shapeString(const cv::Mat& tensor) {
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

cv::Mat MPPoseEstimator::makeNhwcBlob(const cv::Mat& rgb_normalized) {
  const int sizes[] = {1, rgb_normalized.rows, rgb_normalized.cols, rgb_normalized.channels()};
  return cv::Mat(4, sizes, CV_32F, const_cast<float*>(rgb_normalized.ptr<float>())).clone();
}

}  // namespace asdun
