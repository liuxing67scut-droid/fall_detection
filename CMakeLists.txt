cmake_minimum_required(VERSION 3.16)

project(fall_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgcodecs imgproc objdetect videoio)
find_package(CURL QUIET)

add_executable(fall_detect
  src/alert/FallAlertClient.cpp
  src/main.cpp
  src/camera/CameraManager.cpp
  src/core/App.cpp
  src/core/MainMenu.cpp
  src/detect/PersonDetector.cpp
  src/fall/FallDetector.cpp
  src/fall/PoseFallDetector.cpp
  src/pose/MPPersonDetector.cpp
  src/pose/MPPoseEstimator.cpp
  src/pose/MoveNetPoseEstimator.cpp
  src/remote/PoseRemoteClient.cpp
  src/tracking/SinglePersonTracker.cpp
  src/ui/Renderer.cpp
)

target_include_directories(fall_detect PRIVATE include)
target_link_libraries(fall_detect PRIVATE ${OpenCV_LIBS})

if(CURL_FOUND)
  target_compile_definitions(fall_detect PRIVATE USE_REMOTE_POSE_CLIENT)
  target_compile_definitions(fall_detect PRIVATE USE_FALL_ALERT_CLIENT)
  if(TARGET CURL::libcurl)
    target_link_libraries(fall_detect PRIVATE CURL::libcurl)
  else()
    target_include_directories(fall_detect PRIVATE ${CURL_INCLUDE_DIRS})
    target_link_libraries(fall_detect PRIVATE ${CURL_LIBRARIES})
  endif()
endif()

if(MSVC)
  target_compile_options(fall_detect PRIVATE /W4)
else()
  target_compile_options(fall_detect PRIVATE -Wall -Wextra -Wpedantic)
endif()
