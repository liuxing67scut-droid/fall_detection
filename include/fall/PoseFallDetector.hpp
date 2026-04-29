#pragma once

#include <cstdint>

#include "fall/FallDetector.hpp"
#include "pose/MoveNetPoseEstimator.hpp"

namespace asdun {

struct PoseFallStatus {
  FallState state = FallState::NoTarget;
  float trunk_angle_deg = 0.0F;
  float vertical_span_ratio = 1.0F;
  float hip_drop_speed = 0.0F;
  float left_knee_angle_deg = 0.0F;
  float right_knee_angle_deg = 0.0F;
  int visible_keypoints = 0;
  bool has_baseline = false;
  float baseline_vertical_span = 0.0F;
  std::uint64_t state_since_ms = 0;
};

class PoseFallDetector {
 public:
  PoseFallDetector(float landmark_score_threshold,
                   int min_visible_keypoints,
                   float upright_trunk_angle_deg_max,
                   float baseline_update_alpha,
                   float suspect_trunk_angle_deg_min,
                   float suspect_span_ratio_max,
                   float suspect_hip_drop_speed_min,
                   float knee_angle_crouch_max,
                   int confirm_hold_ms,
                   float recover_trunk_angle_deg_max,
                   float recover_span_ratio_min,
                   int recover_hold_ms);

  PoseFallStatus update(const PoseFrameResult& pose_result, std::uint64_t ts_ms);
  const PoseFallStatus& current() const { return status_; }
  void reset();

 private:
  void setState(FallState state, std::uint64_t ts_ms);

  float landmark_score_threshold_ = 0.25F;
  int min_visible_keypoints_ = 10;
  float upright_trunk_angle_deg_max_ = 25.0F;
  float baseline_update_alpha_ = 0.08F;
  float suspect_trunk_angle_deg_min_ = 55.0F;
  float suspect_span_ratio_max_ = 0.62F;
  float suspect_hip_drop_speed_min_ = 0.35F;
  float knee_angle_crouch_max_ = 115.0F;
  int confirm_hold_ms_ = 500;
  float recover_trunk_angle_deg_max_ = 30.0F;
  float recover_span_ratio_min_ = 0.78F;
  int recover_hold_ms_ = 800;

  float baseline_vertical_span_ = 0.0F;
  bool prev_hip_valid_ = false;
  cv::Point2f prev_hip_center_{};
  std::uint64_t prev_ts_ms_ = 0;
  std::uint64_t suspect_since_ms_ = 0;
  std::uint64_t recover_since_ms_ = 0;
  PoseFallStatus status_{};
};

}  // namespace asdun
