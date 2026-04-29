#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "pose/MPPersonDetector.hpp"
#include "pose/MoveNetPoseEstimator.hpp"

namespace asdun {

class MPPoseEstimator {
 public:
  MPPoseEstimator(std::string model_path, float score_threshold);

  bool init();
  PoseFrameResult estimate(const cv::Mat& frame_bgr, const MPPersonRegion& person_region);
  std::string statusMessage() const { return status_message_; }
  bool modelReady() const { return model_ready_; }

 private:
  static std::string shapeString(const cv::Mat& tensor);
  static cv::Mat makeNhwcBlob(const cv::Mat& rgb_normalized);

  std::string model_path_;
  float score_threshold_ = 0.2F;
  cv::Size input_size_{256, 256};
  cv::dnn::Net net_;
  bool model_ready_ = false;
  std::string status_message_{"pose=not_initialized"};
  std::string last_output_shapes_{"unknown"};
};

}  // namespace asdun
