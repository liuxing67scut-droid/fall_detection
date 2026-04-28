#include "tracking/SinglePersonTracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace asdun {

namespace {

float clampAlpha(float alpha) {
  return std::clamp(alpha, 0.0F, 1.0F);
}

}  // namespace

SinglePersonTracker::SinglePersonTracker(float new_track_conf_threshold,
                                         float hold_track_conf_threshold,
                                         float match_iou_threshold,
                                         float max_center_distance_ratio,
                                         int confirm_frames,
                                         int max_missing_frames,
                                         float smooth_alpha)
    : new_track_conf_threshold_(new_track_conf_threshold),
      hold_track_conf_threshold_(hold_track_conf_threshold),
      match_iou_threshold_(match_iou_threshold),
      max_center_distance_ratio_(max_center_distance_ratio),
      confirm_frames_(std::max(confirm_frames, 1)),
      max_missing_frames_(std::max(max_missing_frames, 0)),
      smooth_alpha_(clampAlpha(smooth_alpha)) {}

TrackedPerson SinglePersonTracker::update(const std::vector<PersonDetection>& detections, std::uint64_t frame_id) {
  if (!state_.has_target) {
    const int new_target_index = selectNewTarget(detections);
    if (new_target_index >= 0) {
      startTrack(detections[static_cast<std::size_t>(new_target_index)], frame_id);
    } else {
      reset();
      state_.last_frame_id = frame_id;
    }
    return state_;
  }

  const int matched_index = matchExistingTarget(detections);
  if (matched_index >= 0) {
    applyMatch(detections[static_cast<std::size_t>(matched_index)], frame_id);
    return state_;
  }

  handleMissing(frame_id);
  if (!state_.has_target) {
    const int new_target_index = selectNewTarget(detections);
    if (new_target_index >= 0) {
      startTrack(detections[static_cast<std::size_t>(new_target_index)], frame_id);
    }
  }
  return state_;
}

void SinglePersonTracker::reset() { state_ = TrackedPerson{}; }

int SinglePersonTracker::selectNewTarget(const std::vector<PersonDetection>& detections) const {
  float best_confidence = new_track_conf_threshold_;
  int best_index = -1;
  for (std::size_t i = 0; i < detections.size(); ++i) {
    if (detections[i].confidence < best_confidence) {
      continue;
    }
    best_confidence = detections[i].confidence;
    best_index = static_cast<int>(i);
  }
  return best_index;
}

int SinglePersonTracker::matchExistingTarget(const std::vector<PersonDetection>& detections) const {
  if (!state_.has_target) {
    return -1;
  }

  float best_score = -1.0F;
  int best_index = -1;
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const PersonDetection& detection = detections[i];
    if (detection.confidence < hold_track_conf_threshold_) {
      continue;
    }

    const float overlap = iou(detection.box, state_.box);
    const float center_ratio = centerDistanceRatio(detection.box, state_.box);
    if (overlap < match_iou_threshold_ && center_ratio > max_center_distance_ratio_) {
      continue;
    }

    const float center_score =
        std::clamp(1.0F - center_ratio / std::max(max_center_distance_ratio_, 1e-6F), 0.0F, 1.0F);
    const float score = overlap * 0.75F + center_score * 0.20F + detection.confidence * 0.05F;
    if (score > best_score) {
      best_score = score;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

void SinglePersonTracker::startTrack(const PersonDetection& detection, std::uint64_t frame_id) {
  state_.has_target = true;
  state_.box = detection.box;
  state_.confidence = detection.confidence;
  state_.is_confirmed = false;
  state_.is_holding = false;
  state_.missing_frames = 0;
  state_.detection_hits = 1;
  state_.last_frame_id = frame_id;
}

void SinglePersonTracker::applyMatch(const PersonDetection& detection, std::uint64_t frame_id) {
  state_.box = smoothRect(state_.box, detection.box, smooth_alpha_);
  state_.confidence = detection.confidence;
  state_.missing_frames = 0;
  state_.is_holding = false;
  state_.detection_hits += 1;
  state_.is_confirmed = state_.detection_hits >= confirm_frames_;
  state_.last_frame_id = frame_id;
}

void SinglePersonTracker::handleMissing(std::uint64_t frame_id) {
  if (!state_.has_target) {
    return;
  }

  state_.missing_frames += 1;
  state_.is_holding = true;
  state_.confidence *= 0.92F;
  state_.detection_hits = std::max(0, state_.detection_hits - 1);
  state_.is_confirmed = state_.detection_hits >= confirm_frames_;
  state_.last_frame_id = frame_id;

  if (state_.missing_frames > max_missing_frames_) {
    reset();
  }
}

float SinglePersonTracker::iou(const cv::Rect& a, const cv::Rect& b) {
  const int left = std::max(a.x, b.x);
  const int top = std::max(a.y, b.y);
  const int right = std::min(a.x + a.width, b.x + b.width);
  const int bottom = std::min(a.y + a.height, b.y + b.height);
  if (right <= left || bottom <= top) {
    return 0.0F;
  }

  const float intersection = static_cast<float>((right - left) * (bottom - top));
  const float union_area = static_cast<float>(a.area() + b.area()) - intersection;
  if (union_area <= 1e-6F) {
    return 0.0F;
  }
  return intersection / union_area;
}

float SinglePersonTracker::centerDistanceRatio(const cv::Rect& a, const cv::Rect& b) {
  const float ax = static_cast<float>(a.x) + static_cast<float>(a.width) * 0.5F;
  const float ay = static_cast<float>(a.y) + static_cast<float>(a.height) * 0.5F;
  const float bx = static_cast<float>(b.x) + static_cast<float>(b.width) * 0.5F;
  const float by = static_cast<float>(b.y) + static_cast<float>(b.height) * 0.5F;
  const float dx = ax - bx;
  const float dy = ay - by;
  const float distance = std::sqrt(dx * dx + dy * dy);
  const float scale = static_cast<float>(std::max({a.width, a.height, b.width, b.height}));
  if (scale <= 1e-6F) {
    return std::numeric_limits<float>::max();
  }
  return distance / scale;
}

cv::Rect SinglePersonTracker::smoothRect(const cv::Rect& old_rect, const cv::Rect& new_rect, float alpha) {
  if (old_rect.width <= 0 || old_rect.height <= 0) {
    return new_rect;
  }

  const float a = clampAlpha(alpha);
  auto blend = [a](int old_value, int new_value) {
    return static_cast<int>(std::lround((1.0F - a) * static_cast<float>(old_value) +
                                        a * static_cast<float>(new_value)));
  };

  cv::Rect blended(blend(old_rect.x, new_rect.x),
                   blend(old_rect.y, new_rect.y),
                   blend(old_rect.width, new_rect.width),
                   blend(old_rect.height, new_rect.height));
  if (blended.width <= 0 || blended.height <= 0) {
    return new_rect;
  }
  return blended;
}

}  // namespace asdun
