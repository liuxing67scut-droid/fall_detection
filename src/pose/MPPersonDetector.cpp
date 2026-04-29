#include "pose/MPPersonDetector.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace asdun {

namespace {

constexpr std::array<int, 5> kStrides = {8, 16, 32, 32, 32};

}  // namespace

MPPersonDetector::MPPersonDetector(std::string model_path, float score_threshold, float nms_threshold, int top_k)
    : model_path_(std::move(model_path)),
      score_threshold_(score_threshold),
      nms_threshold_(nms_threshold),
      top_k_(std::max(top_k, 1)) {}

bool MPPersonDetector::init() {
  model_ready_ = false;
  last_output_shape_ = "unknown";
  if (model_path_.empty()) {
    status_message_ = "persondet=model_path_empty";
    return true;
  }

  try {
    net_ = cv::dnn::readNet(model_path_);
    if (net_.empty()) {
      status_message_ = "persondet=load_failed_empty_net";
      return true;
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    anchors_ = generateAnchors();
    model_ready_ = true;
    status_message_ = "persondet=model_loaded";
    return true;
  } catch (const cv::Exception& e) {
    std::cerr << "[Pose/MPPersonDet] failed to load model: " << model_path_ << std::endl;
    std::cerr << "[Pose/MPPersonDet] OpenCV exception: " << e.what() << std::endl;
    status_message_ = "persondet=load_failed";
    return true;
  }
}

std::vector<MPPersonRegion> MPPersonDetector::detect(const cv::Mat& frame_bgr) {
  std::vector<MPPersonRegion> regions;
  if (!model_ready_ || frame_bgr.empty()) {
    return regions;
  }

  try {
    const auto [input_blob, pad_bias] = preprocess(frame_bgr);
    net_.setInput(input_blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.size() < 2) {
      status_message_ = "persondet=unexpected_output_count";
      return regions;
    }

    last_output_shape_ = shapeString(outputs[0]) + "," + shapeString(outputs[1]);

    cv::Mat score = outputs[1].reshape(1, static_cast<int>(outputs[1].total()));
    cv::Mat box_land_delta = outputs[0].reshape(1, outputs[0].size[1]);
    if (box_land_delta.rows != anchors_.rows || score.rows != anchors_.rows) {
      status_message_ = "persondet=shape_mismatch";
      return regions;
    }

    const double scale = static_cast<double>(std::max(frame_bgr.cols, frame_bgr.rows));

    struct Candidate {
      cv::Rect2f box;
      std::array<cv::Point2f, 4> aux_keypoints{};
      float score = 0.0F;
    };

    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(anchors_.rows));

    for (int idx = 0; idx < anchors_.rows; ++idx) {
      const float raw_score = score.at<float>(idx, 0);
      const float prob = sigmoid(std::clamp(raw_score, -100.0F, 100.0F));
      if (prob < score_threshold_) {
        continue;
      }

      const float anchor_x = anchors_.at<float>(idx, 0);
      const float anchor_y = anchors_.at<float>(idx, 1);

      const float cx = (box_land_delta.at<float>(idx, 0) / static_cast<float>(input_size_.width) + anchor_x) *
                           static_cast<float>(scale) -
                       static_cast<float>(pad_bias.width);
      const float cy = (box_land_delta.at<float>(idx, 1) / static_cast<float>(input_size_.height) + anchor_y) *
                           static_cast<float>(scale) -
                       static_cast<float>(pad_bias.height);
      const float w = (box_land_delta.at<float>(idx, 2) / static_cast<float>(input_size_.width)) *
                      static_cast<float>(scale);
      const float h = (box_land_delta.at<float>(idx, 3) / static_cast<float>(input_size_.height)) *
                      static_cast<float>(scale);

      Candidate candidate;
      candidate.score = prob;
      candidate.box = cv::Rect2f(cx - w * 0.5F, cy - h * 0.5F, w, h);

      for (int keypoint = 0; keypoint < 4; ++keypoint) {
        const int offset = 4 + keypoint * 2;
        candidate.aux_keypoints[static_cast<std::size_t>(keypoint)] = cv::Point2f(
            (box_land_delta.at<float>(idx, offset + 0) / static_cast<float>(input_size_.width) + anchor_x) *
                    static_cast<float>(scale) -
                static_cast<float>(pad_bias.width),
            (box_land_delta.at<float>(idx, offset + 1) / static_cast<float>(input_size_.height) + anchor_y) *
                    static_cast<float>(scale) -
                static_cast<float>(pad_bias.height));
      }

      candidates.push_back(candidate);
      nms_boxes.emplace_back(cvRound(candidate.box.x), cvRound(candidate.box.y), cvRound(candidate.box.width),
                             cvRound(candidate.box.height));
      nms_scores.push_back(candidate.score);
    }

    if (candidates.empty()) {
      status_message_ = "persondet=no_person";
      return regions;
    }

    std::vector<int> keep_indices;
    cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_, keep_indices, 1.0F, top_k_);

    for (int keep_index : keep_indices) {
      if (keep_index < 0 || keep_index >= static_cast<int>(candidates.size())) {
        continue;
      }
      MPPersonRegion region;
      region.box = candidates[static_cast<std::size_t>(keep_index)].box;
      region.aux_keypoints = candidates[static_cast<std::size_t>(keep_index)].aux_keypoints;
      region.score = candidates[static_cast<std::size_t>(keep_index)].score;
      regions.push_back(region);
    }

    if (!regions.empty()) {
      status_message_ = "persondet=model_ready shape=" + last_output_shape_;
    } else {
      status_message_ = "persondet=no_person_after_nms";
    }
    return regions;
  } catch (const cv::Exception& e) {
    std::cerr << "[Pose/MPPersonDet] forward failed for model: " << model_path_ << std::endl;
    std::cerr << "[Pose/MPPersonDet] OpenCV exception: " << e.what() << std::endl;
    model_ready_ = false;
    status_message_ = "persondet=forward_failed";
    return regions;
  }
}

cv::Mat MPPersonDetector::generateAnchors() {
  std::vector<cv::Point2f> anchor_points;
  anchor_points.reserve(2254);

  int layer_id = 0;
  while (layer_id < static_cast<int>(kStrides.size())) {
    const int stride = kStrides[static_cast<std::size_t>(layer_id)];
    int last_same_stride_layer = layer_id;
    while (last_same_stride_layer < static_cast<int>(kStrides.size()) &&
           kStrides[static_cast<std::size_t>(last_same_stride_layer)] == stride) {
      ++last_same_stride_layer;
    }

    const int anchors_per_cell = (last_same_stride_layer - layer_id) * 2;
    const int feature_h = static_cast<int>(std::ceil(static_cast<float>(224) / static_cast<float>(stride)));
    const int feature_w = static_cast<int>(std::ceil(static_cast<float>(224) / static_cast<float>(stride)));

    for (int y = 0; y < feature_h; ++y) {
      for (int x = 0; x < feature_w; ++x) {
        const float x_center = (static_cast<float>(x) + 0.5F) / static_cast<float>(feature_w);
        const float y_center = (static_cast<float>(y) + 0.5F) / static_cast<float>(feature_h);
        for (int repeat = 0; repeat < anchors_per_cell; ++repeat) {
          anchor_points.emplace_back(x_center, y_center);
        }
      }
    }

    layer_id = last_same_stride_layer;
  }

  cv::Mat anchors(static_cast<int>(anchor_points.size()), 2, CV_32F);
  for (int index = 0; index < static_cast<int>(anchor_points.size()); ++index) {
    anchors.at<float>(index, 0) = anchor_points[static_cast<std::size_t>(index)].x;
    anchors.at<float>(index, 1) = anchor_points[static_cast<std::size_t>(index)].y;
  }
  return anchors;
}

float MPPersonDetector::sigmoid(float value) {
  return 1.0F / (1.0F + std::exp(-value));
}

std::string MPPersonDetector::shapeString(const cv::Mat& tensor) {
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

std::pair<cv::Mat, cv::Size> MPPersonDetector::preprocess(const cv::Mat& frame_bgr) const {
  cv::dnn::Image2BlobParams params;
  params.datalayout = cv::dnn::DNN_LAYOUT_NCHW;
  params.ddepth = CV_32F;
  params.mean = cv::Scalar::all(127.5);
  params.scalefactor = cv::Scalar::all(1.0 / 127.5);
  params.size = input_size_;
  params.swapRB = true;
  params.paddingmode = cv::dnn::DNN_PMODE_LETTERBOX;

  const double ratio =
      std::min(input_size_.height / static_cast<double>(frame_bgr.rows), input_size_.width / static_cast<double>(frame_bgr.cols));
  cv::Size pad_bias(0, 0);
  if (frame_bgr.rows != input_size_.height || frame_bgr.cols != input_size_.width) {
    const cv::Size ratio_size(static_cast<int>(frame_bgr.cols * ratio), static_cast<int>(frame_bgr.rows * ratio));
    const int pad_h = input_size_.height - ratio_size.height;
    const int pad_w = input_size_.width - ratio_size.width;
    pad_bias.width = static_cast<int>((pad_w / 2) / ratio);
    pad_bias.height = static_cast<int>((pad_h / 2) / ratio);
  }
  return {cv::dnn::blobFromImageWithParams(frame_bgr, params), pad_bias};
}

}  // namespace asdun
