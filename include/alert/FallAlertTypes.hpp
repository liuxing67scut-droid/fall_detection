#pragma once

#include <cstdint>
#include <string>

namespace asdun {

struct FallAlertClientConfig {
  bool enabled{false};
  std::string base_url{"http://100.87.247.58:9000"};
  std::string event_path{"/api/events/fall"};
  std::string device_id{"pi-01"};
  std::string device_token{};
  int connect_timeout_ms{1000};
  int timeout_ms{3000};
  int cooldown_ms{30000};
  int jpeg_quality{85};
  bool debug{false};
};

struct FallAlertEvent {
  std::string source_device{"pi-01"};
  std::uint64_t frame_id{0};
  std::uint64_t ts_ms{0};
  std::string mode{"unknown"};
  std::string fall_state{"fall_detected"};
  std::string message{"Possible fall detected in the scene. Please check immediately."};
  double fps{0.0};
};

}  // namespace asdun
