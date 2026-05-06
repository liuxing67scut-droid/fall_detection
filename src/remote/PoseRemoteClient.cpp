#include "remote/PoseRemoteClient.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#ifdef USE_REMOTE_POSE_CLIENT
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

std::optional<bool> extractBool(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t value_pos = json.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos) {
    return std::nullopt;
  }
  if (json.compare(value_pos, 4, "true") == 0) {
    return true;
  }
  if (json.compare(value_pos, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> extractUint64(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t value_pos = json.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos) {
    return std::nullopt;
  }
  char* end = nullptr;
  const unsigned long long value = std::strtoull(json.c_str() + value_pos, &end, 10);
  if (end == json.c_str() + value_pos) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(value);
}

std::optional<float> extractFloat(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t value_pos = json.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos || json.compare(value_pos, 4, "null") == 0) {
    return std::nullopt;
  }
  char* end = nullptr;
  const float value = std::strtof(json.c_str() + value_pos, &end);
  if (end == json.c_str() + value_pos) {
    return std::nullopt;
  }
  return value;
}

std::string extractArray(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const std::size_t key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return {};
  }
  const std::size_t open = json.find('[', key_pos + needle.size());
  if (open == std::string::npos) {
    return {};
  }
  int depth = 0;
  for (std::size_t i = open; i < json.size(); ++i) {
    if (json[i] == '[') {
      ++depth;
    } else if (json[i] == ']') {
      --depth;
      if (depth == 0) {
        return json.substr(open, i - open + 1);
      }
    }
  }
  return {};
}

std::vector<RemotePoseKeypoint> extractKeypoints(const std::string& json) {
  std::vector<RemotePoseKeypoint> keypoints;
  const std::string array = extractArray(json, "keypoints");
  if (array.empty()) {
    return keypoints;
  }

  std::size_t cursor = 0;
  while (cursor < array.size()) {
    const std::size_t object_open = array.find('{', cursor);
    if (object_open == std::string::npos) {
      break;
    }
    int depth = 0;
    std::size_t object_close = std::string::npos;
    for (std::size_t i = object_open; i < array.size(); ++i) {
      if (array[i] == '{') {
        ++depth;
      } else if (array[i] == '}') {
        --depth;
        if (depth == 0) {
          object_close = i;
          break;
        }
      }
    }
    if (object_close == std::string::npos) {
      break;
    }

    const std::string object = array.substr(object_open, object_close - object_open + 1);
    RemotePoseKeypoint keypoint;
    keypoint.id = static_cast<int>(extractUint64(object, "id").value_or(0));
    keypoint.x = extractFloat(object, "x").value_or(0.0F);
    keypoint.y = extractFloat(object, "y").value_or(0.0F);
    keypoint.score = extractFloat(object, "score").value_or(0.0F);
    keypoints.push_back(std::move(keypoint));
    cursor = object_close + 1;
  }

  return keypoints;
}

#ifdef USE_REMOTE_POSE_CLIENT
std::size_t writeStringCallback(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
  auto* out = static_cast<std::string*>(userdata);
  const std::size_t bytes = size * nmemb;
  out->append(ptr, bytes);
  return bytes;
}
#endif

}  // namespace

PoseRemoteClient::PoseRemoteClient(PoseRemoteClientConfig config) : config_(std::move(config)) {}

PoseRemoteClient::~PoseRemoteClient() { stop(); }

bool PoseRemoteClient::start() {
#ifndef USE_REMOTE_POSE_CLIENT
  std::lock_guard<std::mutex> lock(mutex_);
  status_message_ = "remote_pose=libcurl_missing";
  std::cerr << "[PoseRemote] libcurl is not available; remote pose mode is disabled." << std::endl;
  return false;
#else
  std::lock_guard<std::mutex> lock(mutex_);
  if (running_) {
    return true;
  }
  curl_global_init(CURL_GLOBAL_DEFAULT);
  running_ = true;
  pending_request_.reset();
  latest_result_.reset();
  last_submit_ts_ms_ = 0;
  healthy_ = false;
  status_message_ = "remote_pose=starting";

  std::string health_response;
  healthy_ = probeHealth(&health_response);
  status_message_ = healthy_ ? "remote_pose=health_ok" : "remote_pose=health_failed";
  if (config_.debug) {
    if (healthy_) {
      std::cout << "[PoseRemote] health ok: " << health_response << std::endl;
    } else {
      std::cerr << "[PoseRemote] health check failed: " << healthUrl() << std::endl;
    }
  }

  worker_ = std::thread(&PoseRemoteClient::workerLoop, this);
  return true;
#endif
}

void PoseRemoteClient::stop() {
#ifdef USE_REMOTE_POSE_CLIENT
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
  curl_global_cleanup();
#endif
}

bool PoseRemoteClient::submit(const cv::Mat& frame_bgr, std::uint64_t frame_id, std::uint64_t ts_ms) {
  if (frame_bgr.empty()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!running_) {
    return false;
  }
  if (last_submit_ts_ms_ > 0 && ts_ms >= last_submit_ts_ms_ &&
      ts_ms - last_submit_ts_ms_ < static_cast<std::uint64_t>(std::max(0, config_.submit_interval_ms))) {
    return false;
  }

  PoseRemoteRequest request;
  request.frame_id = frame_id;
  request.ts_ms = ts_ms;
  request.frame_bgr = frame_bgr.clone();
  pending_request_ = std::move(request);
  last_submit_ts_ms_ = ts_ms;
  cv_.notify_one();
  return true;
}

std::optional<RemotePoseResponse> PoseRemoteClient::consumeLatest() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!latest_result_.has_value()) {
    return std::nullopt;
  }
  auto result = latest_result_;
  latest_result_.reset();
  return result;
}

std::string PoseRemoteClient::statusMessage() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return status_message_;
}

bool PoseRemoteClient::healthy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return healthy_;
}

bool PoseRemoteClient::probeHealth(std::string* response) const {
#ifndef USE_REMOTE_POSE_CLIENT
  (void)response;
  return false;
#else
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return false;
  }

  std::string body;
  const std::string url = healthUrl();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(500, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(curl);

  if (response != nullptr) {
    *response = body;
  }
  return code == CURLE_OK && http_code >= 200 && http_code < 300 &&
         extractBool(body, "ok").value_or(false);
#endif
}

bool PoseRemoteClient::analyze(const PoseRemoteRequest& request, RemotePoseResponse* out) const {
  if (out == nullptr) {
    return false;
  }

#ifndef USE_REMOTE_POSE_CLIENT
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

  std::string response;
  curl_mime* mime = curl_mime_init(curl);

  auto addTextPart = [mime](const char* name, const std::string& value) {
    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, name);
    curl_mime_data(part, value.c_str(), CURL_ZERO_TERMINATED);
  };

  addTextPart("frame_id", std::to_string(request.frame_id));
  addTextPart("ts_ms", std::to_string(request.ts_ms));

  curl_mimepart* image_part = curl_mime_addpart(mime);
  curl_mime_name(image_part, "image");
  curl_mime_filename(image_part, "frame.jpg");
  curl_mime_type(image_part, "image/jpeg");
  curl_mime_data(image_part, reinterpret_cast<const char*>(jpg.data()), jpg.size());

  const std::string url = analyzeUrl();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, static_cast<long>(std::max(1, config_.connect_timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(std::max(1, config_.timeout_ms)));
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

  const CURLcode code = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_mime_free(mime);
  curl_easy_cleanup(curl);

  if (code != CURLE_OK || http_code < 200 || http_code >= 300) {
    if (config_.debug) {
      std::cerr << "[PoseRemote] analyze failed frame=" << request.frame_id
                << " curl=" << curl_easy_strerror(code)
                << " http=" << http_code << std::endl;
    }
    return false;
  }

  out->ok = extractBool(response, "ok").value_or(false);
  out->frame_id = extractUint64(response, "frame_id").value_or(request.frame_id);
  out->ts_ms = extractUint64(response, "ts_ms").value_or(request.ts_ms);
  out->latency_ms = static_cast<double>(extractFloat(response, "latency_ms").value_or(0.0F));
  out->pose_score = extractFloat(response, "pose_score").value_or(0.0F);
  out->keypoints = extractKeypoints(response);

  std::ostringstream status;
  status << "remote_pose=" << (out->ok ? "analyze_ok" : "analyze_not_ok")
         << " latency_ms=" << static_cast<int>(out->latency_ms + 0.5)
         << " keypoints=" << out->keypoints.size();
  out->status_message = status.str();

  if (config_.debug) {
    std::cout << "[PoseRemote] analyze ok frame=" << out->frame_id
              << " keypoints=" << out->keypoints.size()
              << " latency_ms=" << out->latency_ms << std::endl;
  }

  return true;
#endif
}

std::string PoseRemoteClient::healthUrl() const { return joinUrl(config_.server_url, config_.health_path); }

std::string PoseRemoteClient::analyzeUrl() const { return joinUrl(config_.server_url, config_.analyze_path); }

void PoseRemoteClient::workerLoop() {
  while (true) {
    PoseRemoteRequest request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return !running_ || pending_request_.has_value(); });
      if (!running_ && !pending_request_.has_value()) {
        break;
      }
      request = std::move(*pending_request_);
      pending_request_.reset();
      status_message_ = "remote_pose=requesting";
    }

    RemotePoseResponse response;
    const bool ok = analyze(request, &response);

    std::lock_guard<std::mutex> lock(mutex_);
    if (ok) {
      latest_result_ = std::move(response);
      status_message_ = latest_result_->status_message;
      healthy_ = true;
    } else {
      status_message_ = "remote_pose=request_failed";
      healthy_ = false;
    }
  }
}

}  // namespace asdun
