# 摔倒检测

[中文](README.md) | [English](README_EN.md)

基于 Raspberry Pi 4B、USB 摄像头和 OpenCV 的室内单人摔倒检测原型系统。项目采用边缘端采集与可视化、PC 端辅助姿态推理的协同方案，在树莓派算力有限的条件下提升姿态检测实时性。

## 项目特点

- 本项目是为设计一款"智能居家机器人"而开发的摔倒检测功能，相关设备架设在机器人上，可在机器人巡逻期间进行识别，检测并报警，适用于多种室内场景，当前限定为单人检测。
- 支持 `YOLO11 + Box` 与 `MediaPipe + Pose` 两类检测路线，便于对比框检测和姿态关键点方案。
- 提供一种 `MediaPipe (PC) + Pose (Pi)` 的协同模式，由树莓派负责采集与显示，PC 端负责运行模型，进行较重的姿态推理。
- 摔倒判断不依赖单帧结果，而是结合人体框比例、姿态角度、髋部下落速度和持续时间进行状态机判断。
- 预留摔倒事件上传接口，可将截图和告警信息发送到私人网站/其他服务端。

## 运行模式

程序启动后会进入主菜单，可选择以下模式：

| 菜单 | 模式 | 说明 |
| --- | --- | --- |
| `1` | `HOG + Box` | 使用 OpenCV HOG 行人检测作为轻量 fallback，主要用于工程链路验证 |
| `2` | `YOLO11 + Box` | 使用 YOLO11 ONNX + OpenCV DNN 检测人体框，并通过跟踪和框几何规则判断摔倒 |
| `3` | `MediaPipe + Pose` | 在树莓派本地运行 MediaPipe/OpenCV Zoo 姿态链路，绘制 33 个关键点并进行姿态规则判断 |
| `4` | `MediaPipe (PC) + Pose (Pi)` | 树莓派负责采集和显示，PC 端 FastAPI 服务负责 MediaPipe Pose 推理，树莓派端继续执行可视化和摔倒判断 |

快捷键：

- `q` / `Esc`：退出
- `s`：保存当前截图到 `data/`
- `i`：显示或隐藏状态信息
- `j`：在姿态模式下显示或隐藏骨架关键点

## 系统架构

```text
USB Camera
    |
    v
CameraManager
    |
    +--> bbox pipeline
    |       PersonDetector(HOG / YOLO11 ONNX)
    |       -> SinglePersonTracker
    |       -> FallDetector
    |       -> Renderer
    |
    +--> pose pipeline
            local:  MPPersonDetector -> MPPoseEstimator
            remote: PoseRemoteClient -> PC Pose Server
            -> PoseFallDetector
            -> Renderer

FallDetector / PoseFallDetector
    -> FallAlertClient
    -> Platform Server POST /api/events/fall
```

核心模块：

- `camera/CameraManager`：摄像头打开、参数配置、后台采集线程和最新帧分发
- `detect/PersonDetector`：YOLO11 ONNX 推理、HOG fallback、单人检测结果筛选
- `tracking/SinglePersonTracker`：单人目标连续跟踪、短时丢框保持和框平滑
- `fall/FallDetector`：基于检测框宽高比、人体高度基线和持续时间的 bbox 摔倒状态机
- `pose/MPPersonDetector`、`pose/MPPoseEstimator`：本地 MediaPipe/OpenCV Zoo 姿态链路
- `remote/PoseRemoteClient`：异步 JPEG 上传、PC 姿态服务请求和关键点结果解析
- `fall/PoseFallDetector`：基于躯干角度、竖向跨度、髋部下落速度、膝角和持续时间的姿态摔倒状态机
- `alert/FallAlertClient`：摔倒触发和恢复时异步上传截图与事件信息
- `ui/Renderer`：预览画面、检测框、骨架和状态文本绘制

## 目录结构

```text
fall_detect/
├── config/              # 运行配置
├── doc/                 # 方案说明
├── include/             # C++ 头文件
├── models/              # 模型文件
├── pc_pose_server/      # PC端服务
├── src/                 # C++ 源码
└── CMakeLists.txt
```

## 依赖

主程序：

- C++17
- CMake 3.16+
- OpenCV，需包含 `core`、`dnn`、`highgui`、`imgcodecs`、`imgproc`、`objdetect`、`videoio`
- libcurl，可选；如果要使用远程姿态模式或摔倒告警上传，则需要安装

PC 姿态服务：

- Python 3.10+
- FastAPI
- Uvicorn
- MediaPipe
- Pillow
- NumPy

Python 依赖见 `pc_pose_server/requirements.txt`。

## 构建与运行

### Raspberry Pi / Linux

 `~/fall_detect` 为示例路径，实际使用时需替换为项目在树莓派上的实际路径。

首次构建：

```bash
cd ~/fall_detect
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

若已配置过 `build/`，日常运行只需重新编译并启动程序：

```bash
cd ~/fall_detect
cmake --build build -j4
./build/fall_detect
```

### PC 姿态服务

模式 `4) MediaPipe (PC) + Pose (Pi)` 需要先启动 PC 端服务。

下面的 Windows 路径是当前开发机示例，使用时需要替换为项目在自己电脑上的实际路径。

首次准备环境：

```powershell
cd "c:\Users\17646\Desktop\fall_detect\pc_pose_server"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

后续启动服务：

```powershell
cd "c:\Users\17646\Desktop\fall_detect\pc_pose_server"
.\.venv\Scripts\Activate.ps1
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

PC 端需要额外准备 MediaPipe Pose Landmarker Lite 模型：

```text
pc_pose_server/models/pose_landmarker_lite.task
```

也可以通过环境变量指定模型位置：

```powershell
$env:PC_POSE_MODEL_PATH="C:\path\to\pose_landmarker_lite.task"
```

健康检查：

```bash
curl http://<pc-ip>:8000/health
```

树莓派端的远程地址在 `config/app.yaml` 中配置：

```yaml
remote_pose_server_url: "http://<pc-ip>:8000"
```

## 参数配置

主要配置文件为 `config/app.yaml`。

常用参数：

- `pipeline_mode`：默认管线，`bbox` 或 `pose`
- `bbox_backend`：bbox 后端，`hog` 或 `yolo11`
- `pose_backend`：姿态后端，`mediapipe` 或 `pc_mediapipe`
- `camera_source`：摄像头来源，常见为 `"0"`
- `frame_width`、`frame_height`、`frame_fps`：摄像头请求参数
- `detector_model_path`：YOLO11 ONNX 模型路径
- `detector_interval`：bbox 检测间隔，用于平衡性能和稳定性
- `mp_persondet_model_path`、`mp_pose_model_path`：本地 MediaPipe/OpenCV Zoo 姿态模型路径
- `remote_pose_server_url`：PC 姿态服务地址
- `fall_alert_enabled`：是否启用摔倒告警上传
- `fall_alert_base_url`、`fall_alert_event_path`：平台告警接口地址

说明：`pipeline_mode`、`bbox_backend` 和 `pose_backend` 会被启动菜单覆盖；细粒度参数仍从 `app.yaml` 读取。

## 摔倒判断逻辑

### bbox 路线

`YOLO11/HOG -> SinglePersonTracker -> FallDetector`：

- 跟踪单个主目标，降低检测框抖动和短时丢框影响
- 用站立状态下的人体框高度维护动态基线
- 结合人体框宽高比、当前高度和基线高度比例判断疑似摔倒
- 只有疑似状态持续达到配置时间后才进入 `fall_detected`
- 恢复站立姿态并持续一段时间后回到 `normal`

### pose 路线

`MediaPipe 33 keypoints -> PoseFallDetector`：

- 统计可见关键点数量
- 计算肩部中心到髋部中心形成的躯干角度
- 维护站立姿态下的身体竖向跨度基线
- 计算髋部向下运动速度
- 使用膝关节角度抑制明显下蹲造成的误报
- 通过 `normal / suspected_fall / fall_detected / no_target` 状态机输出结果

## 摔倒报警上传

主程序已经接入 Pi 端上传客户端 `FallAlertClient`。当状态从非 `fall_detected` 首次进入 `fall_detected` 时，上传一次 `raise` 事件；当状态从 `fall_detected` 恢复到 `normal` 时，上传一次 `clear` 事件。

推荐平台接口：

```http
POST /api/events/fall
Content-Type: multipart/form-data
```

上传字段包括：

- `source_device`
- `frame_id`
- `ts_ms`
- `mode`
- `alert_action`
- `fall_state`
- `message`
- `fps`
- `image`

更完整的联动说明见 `doc/摔倒报警上传联动方案.md`。

## 说明文档

- `doc/摔倒检测项目方案草案_v1.md`：项目阶段规划、技术路线和当前结论
- `doc/摔倒报警上传联动方案.md`：摔倒告警上传接口和联调流程
- `pc_pose_server/README.md`：PC 姿态服务运行说明
