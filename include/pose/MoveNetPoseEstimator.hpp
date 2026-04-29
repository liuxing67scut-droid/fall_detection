#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace asdun {

struct PoseKeypoint {
  std::string name;
  cv::Point2f point;
  float score = 0.0F;
};

struct PoseFrameResult {
  bool has_pose = false;
  std::vector<PoseKeypoint> keypoints;
};

class MoveNetPoseEstimator {
 public:
  MoveNetPoseEstimator(std::string model_path,
                       int input_width,
                       int input_height,
                       float keypoint_score_threshold);

  bool init();
  PoseFrameResult estimate(const cv::Mat& frame_bgr);
  std::string statusMessage() const { return status_message_; }
  bool modelReady() const { return model_ready_; }

 private:
  static std::string shapeString(const cv::Mat& tensor);

  std::string model_path_;
  int input_width_ = 192;
  int input_height_ = 192;
  float keypoint_score_threshold_ = 0.20F;
  cv::dnn::Net net_;
  bool model_ready_ = false;
  std::string status_message_{"pose=not_initialized"};
  std::string last_output_shape_{"unknown"};
};

}  // namespace asdun
