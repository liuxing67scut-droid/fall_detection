#include "detect/PersonDetector.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <utility>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace asdun {

namespace {

cv::Rect clampRect(const cv::Rect& rect, const cv::Size& image_size) {
  return rect & cv::Rect(0, 0, image_size.width, image_size.height);
}

cv::Rect toRectFromCxCywh(float cx, float cy, float width, float height) {
  const int x = static_cast<int>(std::round(cx - width * 0.5F));
  const int y = static_cast<int>(std::round(cy - height * 0.5F));
  const int w = static_cast<int>(std::round(width));
  const int h = static_cast<int>(std::round(height));
  return {x, y, w, h};
}

}  // namespace

PersonDetector::PersonDetector(std::string model_path,
                               int input_width,
                               int input_height,
                               float confidence_threshold,
                               float nms_threshold,
                               int person_class_id,
                               bool swap_rb,
                               bool use_hog_fallback,
                               int max_detections)
    : model_path_(std::move(model_path)),
      input_width_(input_width > 0 ? input_width : 640),
      input_height_(input_height > 0 ? input_height : 640),
      confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold),
      person_class_id_(person_class_id),
      swap_rb_(swap_rb),
      use_hog_fallback_(use_hog_fallback),
      max_detections_(max_detections > 0 ? max_detections : 1) {}

bool PersonDetector::init() {
  dnn_ready_ = false;
  hog_ready_ = false;
  dnn_runtime_failed_ = false;
  status_message_ = "detector not initialized";

  if (!model_path_.empty() && std::filesystem::exists(model_path_)) {
    try {
      net_ = cv::dnn::readNet(model_path_);
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      dnn_ready_ = true;
      status_message_ = "detector=dnn";
    } catch (const cv::Exception& ex) {
      std::cerr << "[Detector] failed to load ONNX model: " << ex.what() << std::endl;
      status_message_ = "detector=model_load_failed";
    }
  } else if (!model_path_.empty()) {
    status_message_ = "detector=model_missing";
  }

  if (use_hog_fallback_) {
    hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    hog_ready_ = true;
    if (!dnn_ready_) {
      status_message_ = "detector=hog_fallback";
    }
  }

  return dnn_ready_ || hog_ready_;
}

std::vector<PersonDetection> PersonDetector::detect(const cv::Mat& frame_bgr) const {
  if (frame_bgr.empty()) {
    return {};
  }
  // If a model can be loaded but crashes during forward(), stop retrying it every frame.
  if (dnn_ready_ && !dnn_runtime_failed_) {
    return detectWithDnn(frame_bgr);
  }
  if (hog_ready_) {
    return detectWithHog(frame_bgr);
  }
  return {};
}

PersonDetector::LetterboxInfo PersonDetector::prepareInput(const cv::Mat& frame_bgr, cv::Mat& padded) const {
  LetterboxInfo info{};
  info.padded_width = input_width_;
  info.padded_height = input_height_;

  const float scale = std::min(static_cast<float>(input_width_) / static_cast<float>(frame_bgr.cols),
                               static_cast<float>(input_height_) / static_cast<float>(frame_bgr.rows));
  const int resized_width = std::max(1, static_cast<int>(std::round(static_cast<float>(frame_bgr.cols) * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::round(static_cast<float>(frame_bgr.rows) * scale)));

  cv::Mat resized;
  cv::resize(frame_bgr, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

  info.pad_x = (input_width_ - resized_width) / 2;
  info.pad_y = (input_height_ - resized_height) / 2;
  info.scale = scale;

  padded = cv::Mat(input_height_, input_width_, frame_bgr.type(), cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(info.pad_x, info.pad_y, resized_width, resized_height)));
  return info;
}

std::vector<PersonDetection> PersonDetector::detectWithDnn(const cv::Mat& frame_bgr) const {
  try {
    cv::Mat padded;
    const LetterboxInfo info = prepareInput(frame_bgr, padded);
    cv::Mat blob = cv::dnn::blobFromImage(
        padded, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(), swap_rb_, false, CV_32F);

    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.empty()) {
      return {};
    }

    cv::Mat output = outputs.front();
    cv::Mat rows;
    if (output.dims == 3) {
      const int rows_count = output.size[1];
      const int cols_count = output.size[2];
      cv::Mat raw(rows_count, cols_count, CV_32F, output.ptr<float>());
      if (rows_count <= cols_count) {
        cv::transpose(raw, rows);
      } else {
        rows = raw.clone();
      }
    } else if (output.dims == 2) {
      rows = output;
    } else {
      return {};
    }

    if (rows.empty() || rows.cols < 5) {
      return {};
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(rows.rows);
    scores.reserve(rows.rows);

    for (int i = 0; i < rows.rows; ++i) {
      const float* data = rows.ptr<float>(i);
      if (rows.cols <= 4 + person_class_id_) {
        continue;
      }

      const float person_score = data[4 + person_class_id_];
      if (person_score < confidence_threshold_) {
        continue;
      }

      const float x = (data[0] - static_cast<float>(info.pad_x)) / info.scale;
      const float y = (data[1] - static_cast<float>(info.pad_y)) / info.scale;
      const float w = data[2] / info.scale;
      const float h = data[3] / info.scale;
      cv::Rect box = clampRect(toRectFromCxCywh(x, y, w, h), frame_bgr.size());
      if (box.width <= 0 || box.height <= 0) {
        continue;
      }

      boxes.push_back(box);
      scores.push_back(person_score);
    }

    std::vector<int> kept_indices;
    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold_, nms_threshold_, kept_indices);

    std::vector<PersonDetection> detections;
    detections.reserve(kept_indices.size());
    for (const int index : kept_indices) {
      detections.push_back(
          PersonDetection{boxes[static_cast<std::size_t>(index)], scores[static_cast<std::size_t>(index)]});
    }

    status_message_ = "detector=dnn";
    return keepTopDetections(std::move(detections));
  } catch (const cv::Exception& ex) {
    std::cerr << "[Detector] DNN inference failed: " << ex.what() << std::endl;
    if (hog_ready_) {
      dnn_runtime_failed_ = true;
      status_message_ = "detector=dnn_failed_hog_fallback";
      std::cerr << "[Detector] disabling DNN for this run and falling back to HOG." << std::endl;
      return detectWithHog(frame_bgr);
    }
    dnn_runtime_failed_ = true;
    status_message_ = "detector=dnn_runtime_failed";
    return {};
  }
}

std::vector<PersonDetection> PersonDetector::detectWithHog(const cv::Mat& frame_bgr) const {
  std::vector<cv::Rect> boxes;
  std::vector<double> weights;
  hog_.detectMultiScale(frame_bgr,
                        boxes,
                        weights,
                        0.0,
                        cv::Size(8, 8),
                        cv::Size(16, 16),
                        1.05,
                        2.0,
                        false);

  std::vector<PersonDetection> detections;
  detections.reserve(boxes.size());
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const float confidence = (i < weights.size()) ? static_cast<float>(weights[i]) : 0.0F;
    detections.push_back(PersonDetection{clampRect(boxes[i], frame_bgr.size()), confidence});
  }
  return keepTopDetections(std::move(detections));
}

std::vector<PersonDetection> PersonDetector::keepTopDetections(std::vector<PersonDetection> detections) const {
  std::sort(detections.begin(), detections.end(), [](const PersonDetection& lhs, const PersonDetection& rhs) {
    if (std::abs(lhs.confidence - rhs.confidence) > 1e-6F) {
      return lhs.confidence > rhs.confidence;
    }
    return lhs.box.area() > rhs.box.area();
  });

  if (static_cast<int>(detections.size()) > max_detections_) {
    detections.resize(static_cast<std::size_t>(max_detections_));
  }
  return detections;
}

}  // namespace asdun
