#pragma once

#include <cstdint>

#include "tracking/SinglePersonTracker.hpp"

namespace asdun {

enum class FallState {
  NoTarget,
  Normal,
  SuspectedFall,
  FallDetected,
};

struct FallStatus {
  FallState state = FallState::NoTarget;
  float aspect_ratio = 0.0F;
  float height_ratio = 1.0F;
  bool has_baseline = false;
  float baseline_height = 0.0F;
  std::uint64_t state_since_ms = 0;
};

class FallDetector {
 public:
  FallDetector(float upright_aspect_ratio_max,
               float baseline_update_alpha,
               float suspect_aspect_ratio_threshold,
               float suspect_height_ratio_threshold,
               int confirm_hold_ms,
               float recover_aspect_ratio_threshold,
               float recover_height_ratio_threshold,
               int recover_hold_ms);

  FallStatus update(const TrackedPerson& tracked_person, std::uint64_t ts_ms);
  const FallStatus& current() const { return status_; }
  void reset();

 private:
  bool isUprightCandidate(const TrackedPerson& tracked_person, float aspect_ratio) const;
  bool isSuspectedFallCandidate(float aspect_ratio, float height_ratio) const;
  bool isRecoveryCandidate(float aspect_ratio, float height_ratio) const;
  void setState(FallState state, std::uint64_t ts_ms);

  float upright_aspect_ratio_max_ = 0.85F;
  float baseline_update_alpha_ = 0.08F;
  float suspect_aspect_ratio_threshold_ = 1.10F;
  float suspect_height_ratio_threshold_ = 0.65F;
  int confirm_hold_ms_ = 700;
  float recover_aspect_ratio_threshold_ = 0.85F;
  float recover_height_ratio_threshold_ = 0.80F;
  int recover_hold_ms_ = 900;

  float baseline_height_ = 0.0F;
  std::uint64_t suspect_since_ms_ = 0;
  std::uint64_t recover_since_ms_ = 0;
  FallStatus status_{};
};

}  // namespace asdun
