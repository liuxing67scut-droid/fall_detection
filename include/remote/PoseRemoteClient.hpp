#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include "remote/PoseRemoteTypes.hpp"

namespace asdun {

class PoseRemoteClient {
 public:
  explicit PoseRemoteClient(PoseRemoteClientConfig config);
  ~PoseRemoteClient();

  PoseRemoteClient(const PoseRemoteClient&) = delete;
  PoseRemoteClient& operator=(const PoseRemoteClient&) = delete;

  bool start();
  void stop();
  bool submit(const cv::Mat& frame_bgr, std::uint64_t frame_id, std::uint64_t ts_ms);
  std::optional<RemotePoseResponse> consumeLatest();
  std::string statusMessage() const;
  bool healthy() const;

 private:
  struct PoseRemoteRequest {
    std::uint64_t frame_id = 0;
    std::uint64_t ts_ms = 0;
    cv::Mat frame_bgr;
  };

  bool probeHealth(std::string* response) const;
  bool analyze(const PoseRemoteRequest& request, RemotePoseResponse* out) const;
  std::string healthUrl() const;
  std::string analyzeUrl() const;
  void workerLoop();

  PoseRemoteClientConfig config_{};
  mutable std::mutex mutex_{};
  std::condition_variable cv_{};
  std::optional<PoseRemoteRequest> pending_request_{};
  std::optional<RemotePoseResponse> latest_result_{};
  std::thread worker_{};
  bool running_{false};
  bool healthy_{false};
  std::uint64_t last_submit_ts_ms_{0};
  std::string status_message_{"remote_pose=not_initialized"};
};

}  // namespace asdun
