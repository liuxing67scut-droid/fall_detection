#include "fall/PoseFallDetector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace asdun {

namespace {

constexpr float kPi = 3.14159265358979323846F;

enum MediaPipePoseIndex : int {
  kNose = 0,
  kLeftShoulder = 11,
  kRightShoulder = 12,
  kLeftHip = 23,
  kRightHip = 24,
  kLeftKnee = 25,
  kRightKnee = 26,
  kLeftAnkle = 27,
  kRightAnkle = 28,
  kLeftHeel = 29,
  kRightHeel = 30,
  kLeftFootIndex = 31,
  kRightFootIndex = 32,
};

bool keypointVisible(const PoseFrameResult& pose_result, int index, float threshold) {
  return index >= 0 && index < static_cast<int>(pose_result.keypoints.size()) &&
         pose_result.keypoints[static_cast<std::size_t>(index)].score >= threshold;
}

std::optional<cv::Point2f> averageVisiblePoints(const PoseFrameResult& pose_result,
                                                const std::vector<int>& indices,
                                                float threshold) {
  cv::Point2f sum(0.0F, 0.0F);
  int count = 0;
  for (int index : indices) {
    if (!keypointVisible(pose_result, index, threshold)) {
      continue;
    }
    sum += pose_result.keypoints[static_cast<std::size_t>(index)].point;
    ++count;
  }
  if (count <= 0) {
    return std::nullopt;
  }
  return sum * (1.0F / static_cast<float>(count));
}

std::optional<cv::Point2f> firstVisiblePoint(const PoseFrameResult& pose_result,
                                             const std::vector<int>& indices,
                                             float threshold) {
  for (int index : indices) {
    if (!keypointVisible(pose_result, index, threshold)) {
      continue;
    }
    return pose_result.keypoints[static_cast<std::size_t>(index)].point;
  }
  return std::nullopt;
}

std::optional<float> jointAngleDeg(const std::optional<cv::Point2f>& a,
                                   const std::optional<cv::Point2f>& b,
                                   const std::optional<cv::Point2f>& c) {
  if (!a.has_value() || !b.has_value() || !c.has_value()) {
    return std::nullopt;
  }
  const cv::Point2f ba = *a - *b;
  const cv::Point2f bc = *c - *b;
  const float norm_ba = std::max(static_cast<float>(cv::norm(ba)), 1e-3F);
  const float norm_bc = std::max(static_cast<float>(cv::norm(bc)), 1e-3F);
  float cosine = (ba.x * bc.x + ba.y * bc.y) / (norm_ba * norm_bc);
  cosine = std::clamp(cosine, -1.0F, 1.0F);
  return std::acos(cosine) * 180.0F / kPi;
}

}  // namespace

PoseFallDetector::PoseFallDetector(float landmark_score_threshold,
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
                                   int recover_hold_ms)
    : landmark_score_threshold_(std::clamp(landmark_score_threshold, 0.0F, 1.0F)),
      min_visible_keypoints_(std::max(min_visible_keypoints, 1)),
      upright_trunk_angle_deg_max_(upright_trunk_angle_deg_max),
      baseline_update_alpha_(std::clamp(baseline_update_alpha, 0.0F, 1.0F)),
      suspect_trunk_angle_deg_min_(suspect_trunk_angle_deg_min),
      suspect_span_ratio_max_(suspect_span_ratio_max),
      suspect_hip_drop_speed_min_(suspect_hip_drop_speed_min),
      knee_angle_crouch_max_(knee_angle_crouch_max),
      confirm_hold_ms_(std::max(confirm_hold_ms, 0)),
      recover_trunk_angle_deg_max_(recover_trunk_angle_deg_max),
      recover_span_ratio_min_(recover_span_ratio_min),
      recover_hold_ms_(std::max(recover_hold_ms, 0)) {}

PoseFallStatus PoseFallDetector::update(const PoseFrameResult& pose_result, std::uint64_t ts_ms) {
  int visible_count = 0;
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();
  for (const auto& keypoint : pose_result.keypoints) {
    if (keypoint.score < landmark_score_threshold_) {
      continue;
    }
    ++visible_count;
    min_y = std::min(min_y, keypoint.point.y);
    max_y = std::max(max_y, keypoint.point.y);
  }

  const auto shoulder_center =
      averageVisiblePoints(pose_result, {kLeftShoulder, kRightShoulder}, landmark_score_threshold_);
  const auto hip_center = averageVisiblePoints(pose_result, {kLeftHip, kRightHip}, landmark_score_threshold_);
  const auto left_hip = firstVisiblePoint(pose_result, {kLeftHip}, landmark_score_threshold_);
  const auto right_hip = firstVisiblePoint(pose_result, {kRightHip}, landmark_score_threshold_);
  const auto left_knee = firstVisiblePoint(pose_result, {kLeftKnee}, landmark_score_threshold_);
  const auto right_knee = firstVisiblePoint(pose_result, {kRightKnee}, landmark_score_threshold_);
  const auto left_ankle =
      firstVisiblePoint(pose_result, {kLeftAnkle, kLeftHeel, kLeftFootIndex}, landmark_score_threshold_);
  const auto right_ankle =
      firstVisiblePoint(pose_result, {kRightAnkle, kRightHeel, kRightFootIndex}, landmark_score_threshold_);

  if (!pose_result.has_pose || visible_count < min_visible_keypoints_ || !shoulder_center.has_value() ||
      !hip_center.has_value() || min_y >= max_y) {
    status_.trunk_angle_deg = 0.0F;
    status_.vertical_span_ratio = 1.0F;
    status_.hip_drop_speed = 0.0F;
    status_.left_knee_angle_deg = 0.0F;
    status_.right_knee_angle_deg = 0.0F;
    status_.visible_keypoints = visible_count;
    status_.has_baseline = baseline_vertical_span_ > 1e-3F;
    status_.baseline_vertical_span = baseline_vertical_span_;
    prev_hip_valid_ = false;
    prev_ts_ms_ = ts_ms;
    suspect_since_ms_ = 0;
    recover_since_ms_ = 0;
    setState(FallState::NoTarget, ts_ms);
    return status_;
  }

  const float current_vertical_span = max_y - min_y;
  const cv::Point2f trunk = *hip_center - *shoulder_center;
  const float trunk_angle_deg =
      std::atan2(std::fabs(trunk.x), std::max(std::fabs(trunk.y), 1e-3F)) * 180.0F / kPi;

  if (trunk_angle_deg <= upright_trunk_angle_deg_max_) {
    if (baseline_vertical_span_ <= 1e-3F) {
      baseline_vertical_span_ = current_vertical_span;
    } else {
      baseline_vertical_span_ =
          (1.0F - baseline_update_alpha_) * baseline_vertical_span_ + baseline_update_alpha_ * current_vertical_span;
    }
  }

  const float vertical_span_ratio =
      baseline_vertical_span_ > 1e-3F ? current_vertical_span / baseline_vertical_span_ : 1.0F;

  float hip_drop_speed = 0.0F;
  if (prev_hip_valid_ && baseline_vertical_span_ > 1e-3F && ts_ms > prev_ts_ms_) {
    const float delta_seconds = static_cast<float>(ts_ms - prev_ts_ms_) / 1000.0F;
    if (delta_seconds > 1e-3F) {
      hip_drop_speed =
          ((*hip_center).y - prev_hip_center_.y) / std::max(baseline_vertical_span_, 1e-3F) / delta_seconds;
    }
  }
  prev_hip_center_ = *hip_center;
  prev_hip_valid_ = true;
  prev_ts_ms_ = ts_ms;

  const auto left_knee_angle = jointAngleDeg(left_hip, left_knee, left_ankle);
  const auto right_knee_angle = jointAngleDeg(right_hip, right_knee, right_ankle);
  const bool crouch_like =
      left_knee_angle.has_value() && right_knee_angle.has_value() &&
      *left_knee_angle <= knee_angle_crouch_max_ && *right_knee_angle <= knee_angle_crouch_max_ &&
      trunk_angle_deg < std::min(80.0F, suspect_trunk_angle_deg_min_ + 15.0F);

  status_.trunk_angle_deg = trunk_angle_deg;
  status_.vertical_span_ratio = vertical_span_ratio;
  status_.hip_drop_speed = hip_drop_speed;
  status_.left_knee_angle_deg = left_knee_angle.value_or(0.0F);
  status_.right_knee_angle_deg = right_knee_angle.value_or(0.0F);
  status_.visible_keypoints = visible_count;
  status_.has_baseline = baseline_vertical_span_ > 1e-3F;
  status_.baseline_vertical_span = baseline_vertical_span_;

  const bool baseline_ready = baseline_vertical_span_ > 1e-3F;
  const bool span_low = baseline_ready && vertical_span_ratio <= suspect_span_ratio_max_;
  const bool strong_horizontal = trunk_angle_deg >= std::min(85.0F, suspect_trunk_angle_deg_min_ + 20.0F);
  const bool posture_candidate =
      trunk_angle_deg >= suspect_trunk_angle_deg_min_ && (span_low || (!baseline_ready && strong_horizontal));
  const bool fast_drop = hip_drop_speed >= suspect_hip_drop_speed_min_;
  const bool entry_candidate = posture_candidate && (fast_drop || strong_horizontal) && !crouch_like;
  const bool maintain_suspect = posture_candidate && !crouch_like;
  const bool recovery_candidate =
      trunk_angle_deg <= recover_trunk_angle_deg_max_ &&
      (!baseline_ready || vertical_span_ratio >= recover_span_ratio_min_);

  if (status_.state == FallState::FallDetected) {
    if (recovery_candidate) {
      if (recover_since_ms_ == 0) {
        recover_since_ms_ = ts_ms;
      } else if (ts_ms >= recover_since_ms_ &&
                 static_cast<int>(ts_ms - recover_since_ms_) >= recover_hold_ms_) {
        recover_since_ms_ = 0;
        setState(FallState::Normal, ts_ms);
      }
    } else {
      recover_since_ms_ = 0;
    }
    return status_;
  }

  if (status_.state == FallState::SuspectedFall) {
    if (maintain_suspect) {
      if (suspect_since_ms_ == 0) {
        suspect_since_ms_ = ts_ms;
      } else if (ts_ms >= suspect_since_ms_ &&
                 static_cast<int>(ts_ms - suspect_since_ms_) >= confirm_hold_ms_) {
        setState(FallState::FallDetected, ts_ms);
      }
    } else {
      suspect_since_ms_ = 0;
      setState(FallState::Normal, ts_ms);
    }
    return status_;
  }

  recover_since_ms_ = 0;
  if (entry_candidate) {
    if (suspect_since_ms_ == 0) {
      suspect_since_ms_ = ts_ms;
    }
    setState(FallState::SuspectedFall, ts_ms);
    return status_;
  }

  suspect_since_ms_ = 0;
  setState(FallState::Normal, ts_ms);
  return status_;
}

void PoseFallDetector::reset() {
  baseline_vertical_span_ = 0.0F;
  prev_hip_valid_ = false;
  prev_hip_center_ = cv::Point2f();
  prev_ts_ms_ = 0;
  suspect_since_ms_ = 0;
  recover_since_ms_ = 0;
  status_ = PoseFallStatus{};
}

void PoseFallDetector::setState(FallState state, std::uint64_t ts_ms) {
  if (status_.state != state) {
    status_.state_since_ms = ts_ms;
  }
  status_.state = state;
}

}  // namespace asdun
