#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "detect/PersonDetector.hpp"

namespace asdun {

struct TrackedPerson {
  bool has_target = false;
  cv::Rect box;
  float confidence = 0.0F;
  bool is_confirmed = false;
  bool is_holding = false;
  int missing_frames = 0;
  int detection_hits = 0;
  std::uint64_t last_frame_id = 0;
};

class SinglePersonTracker {
 public:
  SinglePersonTracker(float new_track_conf_threshold,
                      float hold_track_conf_threshold,
                      float match_iou_threshold,
                      float max_center_distance_ratio,
                      int confirm_frames,
                      int max_missing_frames,
                      float smooth_alpha);

  TrackedPerson update(const std::vector<PersonDetection>& detections, std::uint64_t frame_id);
  const TrackedPerson& current() const { return state_; }
  void reset();

 private:
  int selectNewTarget(const std::vector<PersonDetection>& detections) const;
  int matchExistingTarget(const std::vector<PersonDetection>& detections) const;
  void startTrack(const PersonDetection& detection, std::uint64_t frame_id);
  void applyMatch(const PersonDetection& detection, std::uint64_t frame_id);
  void handleMissing(std::uint64_t frame_id);
  static float iou(const cv::Rect& a, const cv::Rect& b);
  static float centerDistanceRatio(const cv::Rect& a, const cv::Rect& b);
  static cv::Rect smoothRect(const cv::Rect& old_rect, const cv::Rect& new_rect, float alpha);

  float new_track_conf_threshold_ = 0.25F;
  float hold_track_conf_threshold_ = 0.10F;
  float match_iou_threshold_ = 0.20F;
  float max_center_distance_ratio_ = 0.45F;
  int confirm_frames_ = 2;
  int max_missing_frames_ = 6;
  float smooth_alpha_ = 0.35F;
  TrackedPerson state_{};
};

}  // namespace asdun
