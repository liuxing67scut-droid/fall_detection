#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <opencv2/core.hpp>

#include "alert/FallAlertTypes.hpp"

namespace asdun {

class FallAlertClient {
 public:
  explicit FallAlertClient(FallAlertClientConfig config);
  ~FallAlertClient();

  FallAlertClient(const FallAlertClient&) = delete;
  FallAlertClient& operator=(const FallAlertClient&) = delete;

  bool start();
  void stop();
  bool enabled() const { return config_.enabled; }
  bool enqueue(const cv::Mat& frame_bgr, const FallAlertEvent& event);
  std::string statusMessage() const;

 private:
  struct PendingRequest {
    FallAlertEvent event{};
    cv::Mat frame_bgr{};
  };

  bool postEvent(const PendingRequest& request) const;
  std::string eventUrl() const;
  void workerLoop();

  FallAlertClientConfig config_{};
  mutable std::mutex mutex_{};
  std::condition_variable cv_{};
  std::deque<PendingRequest> pending_requests_{};
  std::thread worker_{};
  bool running_{false};
  std::uint64_t last_accept_ts_ms_{0};
  std::string status_message_{"fall_alert=disabled"};
};

}  // namespace asdun
