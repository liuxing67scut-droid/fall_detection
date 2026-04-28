#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/objdetect.hpp>

namespace asdun {

struct PersonDetection {
  cv::Rect box;
  float confidence = 0.0F;
};

class PersonDetector {
 public:
  PersonDetector(std::string model_path,
                 int input_width,
                 int input_height,
                 float confidence_threshold,
                 float nms_threshold,
                 int person_class_id,
                 bool swap_rb,
                 bool use_hog_fallback,
                 int max_detections);

  bool init();
  std::vector<PersonDetection> detect(const cv::Mat& frame_bgr) const;

  bool usingDnn() const { return dnn_ready_; }
  bool usingHogFallback() const { return hog_ready_; }
  std::string statusMessage() const { return status_message_; }

 private:
  struct LetterboxInfo {
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
    int padded_width = 0;
    int padded_height = 0;
  };

  LetterboxInfo prepareInput(const cv::Mat& frame_bgr, cv::Mat& padded) const;
  std::vector<PersonDetection> detectWithDnn(const cv::Mat& frame_bgr) const;
  std::vector<PersonDetection> detectWithHog(const cv::Mat& frame_bgr) const;
  std::vector<PersonDetection> keepTopDetections(std::vector<PersonDetection> detections) const;

  std::string model_path_;
  int input_width_ = 640;
  int input_height_ = 640;
  float confidence_threshold_ = 0.45F;
  float nms_threshold_ = 0.45F;
  int person_class_id_ = 0;
  bool swap_rb_ = true;
  bool use_hog_fallback_ = true;
  int max_detections_ = 1;

  bool dnn_ready_ = false;
  bool hog_ready_ = false;
  mutable bool dnn_runtime_failed_ = false;
  mutable std::string status_message_{"detector not initialized"};

  mutable cv::dnn::Net net_;
  cv::HOGDescriptor hog_;
};

}  // namespace asdun
