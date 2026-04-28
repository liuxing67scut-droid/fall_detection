#include "fall/FallDetector.hpp"

#include <algorithm>

namespace asdun {

FallDetector::FallDetector(float upright_aspect_ratio_max,
                           float baseline_update_alpha,
                           float suspect_aspect_ratio_threshold,
                           float suspect_height_ratio_threshold,
                           int confirm_hold_ms,
                           float recover_aspect_ratio_threshold,
                           float recover_height_ratio_threshold,
                           int recover_hold_ms)
    : upright_aspect_ratio_max_(upright_aspect_ratio_max),
      baseline_update_alpha_(std::clamp(baseline_update_alpha, 0.0F, 1.0F)),
      suspect_aspect_ratio_threshold_(suspect_aspect_ratio_threshold),
      suspect_height_ratio_threshold_(suspect_height_ratio_threshold),
      confirm_hold_ms_(std::max(confirm_hold_ms, 0)),
      recover_aspect_ratio_threshold_(recover_aspect_ratio_threshold),
      recover_height_ratio_threshold_(recover_height_ratio_threshold),
      recover_hold_ms_(std::max(recover_hold_ms, 0)) {}

FallStatus FallDetector::update(const TrackedPerson& tracked_person, std::uint64_t ts_ms) {
  if (!tracked_person.has_target || tracked_person.box.width <= 0 || tracked_person.box.height <= 0) {
    status_.aspect_ratio = 0.0F;
    status_.height_ratio = 1.0F;
    status_.has_baseline = baseline_height_ > 1e-3F;
    status_.baseline_height = baseline_height_;
    suspect_since_ms_ = 0;
    recover_since_ms_ = 0;
    setState(FallState::NoTarget, ts_ms);
    return status_;
  }

  const float aspect_ratio =
      static_cast<float>(tracked_person.box.width) / std::max(static_cast<float>(tracked_person.box.height), 1.0F);

  if (isUprightCandidate(tracked_person, aspect_ratio)) {
    const float current_height = static_cast<float>(tracked_person.box.height);
    if (baseline_height_ <= 1e-3F) {
      baseline_height_ = current_height;
    } else {
      baseline_height_ = (1.0F - baseline_update_alpha_) * baseline_height_ + baseline_update_alpha_ * current_height;
    }
  }

  const float height_ratio =
      (baseline_height_ > 1e-3F) ? static_cast<float>(tracked_person.box.height) / baseline_height_ : 1.0F;

  status_.aspect_ratio = aspect_ratio;
  status_.height_ratio = height_ratio;
  status_.has_baseline = baseline_height_ > 1e-3F;
  status_.baseline_height = baseline_height_;

  const bool suspected = isSuspectedFallCandidate(aspect_ratio, height_ratio);
  const bool recovered = isRecoveryCandidate(aspect_ratio, height_ratio);

  if (suspected) {
    recover_since_ms_ = 0;
    if (status_.state != FallState::SuspectedFall && status_.state != FallState::FallDetected) {
      suspect_since_ms_ = ts_ms;
      setState(FallState::SuspectedFall, ts_ms);
    } else if (status_.state == FallState::SuspectedFall && suspect_since_ms_ > 0 &&
               ts_ms >= suspect_since_ms_ &&
               static_cast<int>(ts_ms - suspect_since_ms_) >= confirm_hold_ms_) {
      setState(FallState::FallDetected, ts_ms);
    }
    return status_;
  }

  suspect_since_ms_ = 0;
  if (status_.state == FallState::FallDetected) {
    if (recovered) {
      if (recover_since_ms_ == 0) {
        recover_since_ms_ = ts_ms;
      } else if (ts_ms >= recover_since_ms_ &&
                 static_cast<int>(ts_ms - recover_since_ms_) >= recover_hold_ms_) {
        setState(FallState::Normal, ts_ms);
        recover_since_ms_ = 0;
      }
    } else {
      recover_since_ms_ = 0;
    }
    return status_;
  }

  recover_since_ms_ = 0;
  setState(FallState::Normal, ts_ms);
  return status_;
}

void FallDetector::reset() {
  baseline_height_ = 0.0F;
  suspect_since_ms_ = 0;
  recover_since_ms_ = 0;
  status_ = FallStatus{};
}

bool FallDetector::isUprightCandidate(const TrackedPerson& tracked_person, float aspect_ratio) const {
  return tracked_person.is_confirmed && !tracked_person.is_holding && aspect_ratio <= upright_aspect_ratio_max_;
}

bool FallDetector::isSuspectedFallCandidate(float aspect_ratio, float height_ratio) const {
  const bool aspect_trigger = aspect_ratio >= suspect_aspect_ratio_threshold_;
  const bool height_trigger = height_ratio <= suspect_height_ratio_threshold_;
  return aspect_trigger && height_trigger;
}

bool FallDetector::isRecoveryCandidate(float aspect_ratio, float height_ratio) const {
  return aspect_ratio <= recover_aspect_ratio_threshold_ && height_ratio >= recover_height_ratio_threshold_;
}

void FallDetector::setState(FallState state, std::uint64_t ts_ms) {
  if (status_.state != state) {
    status_.state_since_ms = ts_ms;
  }
  status_.state = state;
}

}  // namespace asdun
