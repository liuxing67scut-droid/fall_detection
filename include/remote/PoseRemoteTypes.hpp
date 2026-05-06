#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace asdun {

struct RemotePoseKeypoint {
  int id = -1;
  float x = 0.0F;
  float y = 0.0F;
  float score = 0.0F;
};

struct RemotePoseResponse {
  bool ok = false;
  std::uint64_t frame_id = 0;
  std::uint64_t ts_ms = 0;
  double latency_ms = 0.0;
  float pose_score = 0.0F;
  std::vector<RemotePoseKeypoint> keypoints;
  std::string status_message = "remote_pose=no_result";
};

struct PoseRemoteClientConfig {
  std::string server_url = "http://100.87.247.58:8000";
  std::string health_path = "/health";
  std::string analyze_path = "/analyze_pose";
  int connect_timeout_ms = 300;
  int timeout_ms = 1500;
  int submit_interval_ms = 220;
  int jpeg_quality = 80;
  bool debug = false;
};

}  // namespace asdun
