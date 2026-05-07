# 摔倒报警上传骨架交付说明

## 1. 这次新增了什么

本次在 `fall_detect` 项目中，先补齐了 **Pi 端的摔倒报警上传骨架**，目标是为后续网站/平台侧实现留好接口。

当前已经完成：

- 详细方案文档：
  - [摔倒报警上传联动方案.md](./摔倒报警上传联动方案.md)
- Pi 端上传骨架：
  - [include/alert/FallAlertTypes.hpp](../include/alert/FallAlertTypes.hpp)
  - [include/alert/FallAlertClient.hpp](../include/alert/FallAlertClient.hpp)
  - [src/alert/FallAlertClient.cpp](../src/alert/FallAlertClient.cpp)
- `App` 内部接入：
  - [include/core/App.hpp](../include/core/App.hpp)
  - [src/core/App.cpp](../src/core/App.cpp)
- 配置项：
  - [config/app.yaml](../config/app.yaml)

## 2. 当前已经打通到哪一步

当前 Pi 端已经具备：

- 在 `bbox` 或 `pose` 路线下监听 `FallState`
- 当状态 **首次切换到** `fall_detected` 时触发一次 `raise` 报警上传
- 当状态从 `fall_detected` 恢复到 `normal` 时触发一次 `clear` 解除上传
- 自动读取当前原始画面并 JPEG 编码
- 通过 `multipart/form-data` 方式准备上传：
  - 文本字段
  - 一张图片
- 支持冷却时间，避免短时间重复刷屏
- 支持状态字符串回显，例如：
  - `fall_alert=ready`
  - `fall_alert=queued`
  - `fall_alert=uploading`
  - `fall_alert=upload_ok`
  - `fall_alert=upload_failed`
  - `fall_alert=cooldown`

## 3. 当前还缺什么

当前 **网站/平台侧接口还没有正式接通**，所以现在这部分只是“Pi 端准备好了”，还不能直接完成整条链路。

后续还需要平台实现同学补齐：

- `POST /api/events/fall`
- 支持 `multipart/form-data`
- 保存上传图片
- 保存摔倒事件记录
- 在网页端展示告警记录

## 4. 推荐先看哪份文档

如果是网站/平台实现同学，建议阅读顺序：

1. [摔倒报警上传联动方案.md](./摔倒报警上传联动方案.md)
2. `reference/` 里队友原有平台相关代码
3. [config/app.yaml](../config/app.yaml) 里的 `fall_alert_*` 配置
4. [src/alert/FallAlertClient.cpp](../src/alert/FallAlertClient.cpp) 里的实际上传字段

## 5. 当前默认行为

当前默认配置是：

- `fall_alert_enabled: true`
- `fall_alert_base_url: "http://8.163.47.15:9000"`

也就是说：

- 当前主工程已经默认指向正式网站地址
- 进入 `fall_detected` 时会触发报警上传
- 恢复到 `normal` 时会触发解除上传

## 6. 交付时最重要的接口约定

Pi 端准备上传的核心字段包括：

- `source_device`
- `frame_id`
- `ts_ms`
- `mode`
- `alert_action`
- `fall_state`
- `message`
- `fps`
- `image`

推荐平台新增接口：

- `POST /api/events/fall`

更细节说明请看：

- [摔倒报警上传联动方案.md](./摔倒报警上传联动方案.md)
