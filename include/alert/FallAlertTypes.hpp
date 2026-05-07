#pragma once

#include <cstdint>
#include <string>

namespace asdun {

struct FallAlertClientConfig {
  bool enabled{true};
  std::string base_url{"http://8.163.47.15:9000"};
  std::string event_path{"/api/events/fall"};
  std::string device_id{"visual-fall-detector"};
  std::string device_token{"vision123456"};
  int connect_timeout_ms{1000};
  int timeout_ms{3000};
  int cooldown_ms{30000};
  int jpeg_quality{85};
  bool debug{false};
};

struct FallAlertEvent {
  std::string source_device{"visual-fall-detector"};
  std::uint64_t frame_id{0};
  std::uint64_t ts_ms{0};
  std::string mode{"unknown"};
  std::string alert_action{"raise"};
  std::string fall_state{"fall_detected"};
  std::string message{"Possible fall detected in the scene. Please check immediately."};
  double fps{0.0};
};

}  // namespace asdun
