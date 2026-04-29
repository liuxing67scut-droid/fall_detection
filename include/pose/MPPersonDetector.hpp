#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace asdun {

struct MPPersonRegion {
  cv::Rect2f box;
  std::array<cv::Point2f, 4> aux_keypoints{};
  float score = 0.0F;

  bool valid() const { return box.width > 0.0F && box.height > 0.0F; }
};

class MPPersonDetector {
 public:
  MPPersonDetector(std::string model_path, float score_threshold, float nms_threshold, int top_k);

  bool init();
  std::vector<MPPersonRegion> detect(const cv::Mat& frame_bgr);
  std::string statusMessage() const { return status_message_; }
  bool modelReady() const { return model_ready_; }

 private:
  static cv::Mat generateAnchors();
  static float sigmoid(float value);
  static std::string shapeString(const cv::Mat& tensor);
  std::pair<cv::Mat, cv::Size> preprocess(const cv::Mat& frame_bgr) const;

  std::string model_path_;
  float score_threshold_ = 0.5F;
  float nms_threshold_ = 0.3F;
  int top_k_ = 1;
  cv::Size input_size_{224, 224};
  cv::dnn::Net net_;
  cv::Mat anchors_;
  bool model_ready_ = false;
  std::string status_message_{"persondet=not_initialized"};
  std::string last_output_shape_{"unknown"};
};

}  // namespace asdun
