#include "alert/FallAlertClient.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#ifdef USE_FALL_ALERT_CLIENT
#include <curl/curl.h>
#endif

namespace asdun {

namespace {

std::string trimTrailingSlash(std::string value) {
  while (!value.empty() && value.back() == '/') {
    value.pop_back();
  }
  return value;
}

std::string joinUrl(const std::string& base_url, const std::string& path) {
  if (path.empty()) {
    return trimTrailingSlash(base_url);
  }
  if (!path.empty() && path.front() == '/') {
    return trimTrailingSlash(base_url) + path;
  }
  return trimTrailingSlash(base_url) + "/" + path;
}

bool isRaiseAction(const std::string& alert_action) {
  std::string lower;
  lower.reserve(alert_action.size());
  for (char ch : alert_action) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return lower == "raise";
}

#ifdef USE_FALL_ALERT_CLIENT
std::size_t discardResponseCallback(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  (void)ptr;
  (void)userdata;
  return size * nmemb;
}
#endif

}  // namespace

FallAlertClient::FallAlertClient(FallAlertClientConfig config) : config_(std::move(config)) {
  if (!config_.enabled) {
    status_message_ = "fall_alert=disabled";
  }
}

FallAlertClient::~FallAlertClient() { stop(); }

bool FallAlertClient::start() {
  if (!config_.enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    status_message_ = "fall_alert=disabled";
    return false;
  }

#ifndef USE_FALL_ALERT_CLIENT
  std::lock_guard<std::mutex> lock(mutex_);
  status_message_ = "fall_alert=libcurl_missing";
  std::cerr << "[FallAlert] libcurl is not available; alert uploading disabled." << std::endl;
  return false;
#else
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return true;
  }
  curl_global_init(CURL_GLOBAL_DEFAULT);
  running_ = true;
  pending_requests_.clear();
  last_accept_ts_ms_ = 0;
  status_message_ = "fall_alert=ready";
  worker_ = std::thread(&FallAlertClient::workerLoop, this);
  return true;
#endif
}

void FallAlertClient::stop() {
#ifdef USE_FALL_ALERT_CLIENT
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
      return;
    }
    running_ = false;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  std::lock_guard<std::mutex> lock(mutex_);
  status_message_ = "fall_alert=stopped";
#endif
}

bool FallAlertClient::enqueue(const cv::Mat& frame_bgr, const FallAlertEvent& event) {
  if (!config_.enabled || frame_bgr.empty()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    status_message_ = "fall_alert=not_running";
    return false;
  }
  const bool apply_cooldown = isRaiseAction(event.alert_action);
  if (apply_cooldown && last_accept_ts_ms_ > 0 && event.ts_ms >= last_accept_ts_ms_ &&
      event.ts_ms - last_accept_ts_ms_ < static_cast<std::uint64_t>(std::max(0, config_.cooldown_ms))) {
    status_message_ = "fall_alert=cooldown";
    return false;
  }

  PendingRequest request;
  request.event = event;
  request.frame_bgr = frame_bgr.clone();
  pending_requests_.push_back(std::move(request));
  last_accept_ts_ms_ = event.ts_ms;
  status_message_ = "fall_alert=queued";
  cv_.notify_one();
  return true;
}

std::string FallAlertClient::statusMessage() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_message_;
}

bool FallAlertClient::postEvent(const PendingRequest& request) const {
#ifndef USE_FALL_ALERT_CLIENT
  (void)request;
  return false;
#else
  std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, std::clamp(config_.jpeg_quality, 30, 100)};
  std::vector<uchar> jpg;
  if (!cv::imencode(".jpg", request.frame_bgr, jpg, params) || jpg.empty()) {
    return false;
  }

  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  curl_mime* mime = curl_mime_init(curl);
  auto addTextPart = [mime](const char* name, const std::string& value) {
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, name);
    curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED);
  };

  addTextPart("source_device", request.event.source_device);
  addTextPart("frame_id", std::to_string(request.event.frame_id));
  addTextPart("ts_ms", std::to_string(request.event.ts_ms));
  addTextPart("mode", request.event.mode);
  addTextPart("alert_action", request.event.alert_action);
  addTextPart("fall_state", request.event.fall_state);
  addTextPart("message", request.event.message);
  addTextPart("fps", std::to_string(request.event.fps));

  curl_mimepart* image_part = curl_mime_addpart(mime);
  curl_mime_name(image_part, "image");
  curl_mime_filename(image_part, "fall_alert.jpg");
  curl_mime_type(image_part, "image/jpeg");
  curl_mime_data(image_part, reinterpret_cast<const char*>(jpg.data()), jpg.size());

  struct curl_slist* headers = nullptr;
  const std::string device_id_header = "X-ASDUN-Device-Id: " + config_.device_id;
  headers = curl_slist_append(headers, device_id_header.c_str());
  if (!config_.device_token.empty()) {
    const std::string token_header = "X-ASDUN-Device-Token: " + config_.device_token;
    headers = curl_slist_append(headers, token_header.c_str());
  }

  curl_easy_setopt(curl, CURLOPT_URL, eventUrl().c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(1, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, discardResponseCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  curl_slist_free_all(headers);
  curl_mime_free(mime);
  curl_easy_cleanup(curl);

  const bool ok = code == CURLE_OK && http_code >= 200 && http_code < 300;
  if (config_.debug) {
    if (ok) {
      std::cout << "[FallAlert] uploaded frame_id=" << request.event.frame_id
                << " ts_ms=" << request.event.ts_ms
                << " mode=" << request.event.mode
                << " fps=" << request.event.fps << std::endl;
    } else {
      std::cerr << "[FallAlert] upload failed curl=" << curl_easy_strerror(code)
                << " http=" << http_code
                << " url=" << eventUrl() << std::endl;
    }
  }
  return ok;
#endif
}

std::string FallAlertClient::eventUrl() const { return joinUrl(config_.base_url, config_.event_path); }

void FallAlertClient::workerLoop() {
#ifdef USE_FALL_ALERT_CLIENT
  while (true) {
    std::optional<PendingRequest> request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return !running_ || !pending_requests_.empty(); });
      if (!running_) {
        break;
      }
      request = std::move(pending_requests_.front());
      pending_requests_.pop_front();
      status_message_ = "fall_alert=uploading";
    }

    bool ok = false;
    if (request.has_value()) {
      ok = postEvent(*request);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      status_message_ = ok ? "fall_alert=upload_ok" : "fall_alert=upload_failed";
    }
  }
#endif
}

}  // namespace asdun
