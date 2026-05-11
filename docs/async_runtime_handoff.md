# 异步推理脚本交接说明

这份文档是给接手者直接使用的，目标是让你快速弄清楚 `examples/async_runtime_loop.py`、`examples/alicia_m_async_runtime.py`、`examples/validate_dataset_async_inference.py` 这三个文件分别负责什么、依赖什么、该怎么跑，以及改动时应该优先看哪里。

如果你只想先记住一句话：

- `examples/async_runtime_loop.py` 是「通用模板」；
- `examples/alicia_m_async_runtime.py` 是「Alicia-M 真机接入版」；
- `examples/validate_dataset_async_inference.py` 是「异步 runtime 的数据集验证版」。

## 先看结论

| 文件 | 角色 | 适合谁用 | 当前状态 |
| --- | --- | --- | --- |
| `examples/async_runtime_loop.py` | 通用控制循环模板 | 要把异步 runtime 接到自己机器 / 自己相机 / 自己机器人上的人 | 需要先补齐相机、状态、动作接口，当前是模板 |
| `examples/alicia_m_async_runtime.py` | Alicia-M 真机控制脚本 | 需要直接驱动 Alicia-M 硬件的人 | 可以直接跑，默认异步，也支持 `--inference-mode sync` |
| `examples/validate_dataset_async_inference.py` | 离线数据集异步验证脚本 | 要检查模型加载、异步队列、动作对齐和误差的人 | 可以直接跑，输出图、CSV 和汇总 |

## 通用前置条件

先在仓库根目录激活环境：

```bash
cd /home/ubuntu/Templates/Inference-SDK
source .venv/bin/activate
```

常见依赖与前提：

- 需要能导入 `inference_sdk`。
- 需要可用的模型目录，或 Hugging Face repo id。
- `act` 支持 `--temporal-ensemble`。
- `smolvla` / `pi0` / `pi05` 支持 `--enable-rtc`。
- 如果要跑 Alicia-M 真机，还需要 `Alicia-M-SDK` 和 OpenCV。

## 1) `examples/async_runtime_loop.py`

### 这个文件做什么

这是最小、最干净的进程内异步控制循环模板。它已经把 runtime 的生命周期处理好了：

1. `load_policy()`
2. `warmup()`
3. `start()`
4. `wait_until_ready()`
5. 循环执行 `step()`
6. 退出时 `stop()`

但它**不负责**你的硬件接入。你必须自己实现这三个函数：

- `read_camera_images()`
- `read_robot_state()`
- `send_robot_action()`

### 什么时候用它

- 你要接一套新的机器人平台。
- 你想复用仓库里的异步 runtime，但不想依赖 Alicia-M 这套硬件接口。
- 你在写上层业务代码，想先拿一个最小可工作的控制框架。

### 直接改哪里

- 把 `read_camera_images()` 接到你的相机管线，返回 `dict[str, np.ndarray]`。
- 把 `read_robot_state()` 接到你的状态读取接口，返回机器人状态向量。
- 把 `send_robot_action()` 接到你的动作下发接口。

### 运行方式

脚本参数和实际 runtime 的最小参数一致：

- `--model-type`
- `--checkpoint-dir`
- `--device`
- `--instruction`
- `--fps`
- `--chunk-size-threshold`
- `--action-chunk-size`
- `--n-action-steps`
- `--temporal-ensemble`
- `--enable-rtc`

示意命令：

```bash
python examples/async_runtime_loop.py \
  --model-type act \
  --checkpoint-dir models/ACT_pick_and_place_v2 \
  --device cuda:0 \
  --fps 30
```

### 需要注意的点

- `get_global_async_runtime()` 返回的是**进程内全局** runtime，不是跨进程共享对象。
- `wait_until_ready(min_queue_size=1)` 没过，说明 warmup / 模型加载 / 采样输入至少有一个环节不稳定。
- 这份模板只负责控制流，真正的 I/O 一定要在你自己的工程里替换掉。

## 2) `examples/alicia_m_async_runtime.py`

### 这个文件做什么

这是 Alicia-M 真机接入脚本。它直接连接 `Alicia-M-SDK`，读取机器人状态和 OpenCV 相机帧，然后把 policy 动作发布给机器人。

它支持两种模式：

- `--inference-mode async`：默认模式，使用 `AsyncInferenceRuntime`。
- `--inference-mode sync`：ACT 抖动明显时可切到同步 `InferenceSDK.predict_action()`。

### 它和模板的区别

和 `examples/async_runtime_loop.py` 相比，这个文件已经把真机需要的东西补好了：

- Alicia-M 机器人连接 / 断开。
- 机器人状态读取。
- 机器人动作发布。
- OpenCV 相机读取。
- 关机时自动 hold 当前姿态。

### 输入约定

- 机器人状态期望是 `[6 个关节, gripper]`。
- 进入 policy 前，6 个关节会从 rad 转成 degree。
- 动作支持 6 维或 7 维：
  - 6 维：只发布关节，夹爪沿用当前值；
  - 7 维：第 7 维是 gripper。

### 关键依赖

- `Alicia-M-SDK`
- `opencv-python`
- 可连接的机器人串口
- 能读到的相机源

如果你没有把 Alicia-M SDK 安装成包，脚本会尝试：

1. 从环境变量 `ALICIA_M_SDK_PATH` 找；
2. 或者找仓库旁边的 `../Alicia-M-SDK`。

### 直接可用的命令

```bash
python examples/alicia_m_async_runtime.py \
  --model-type act \
  --checkpoint-dir models/ACT_pick_and_place_v2 \
  --device cuda:0 \
  --port /dev/ttyACM0 \
  --camera head=0 \
  --camera wrist=2 \
  --fps 30
```

如果 ACT 的异步队列抖动比较明显，可以试同步模式：

```bash
python examples/alicia_m_async_runtime.py \
  --model-type act \
  --checkpoint-dir models/ACT_pick_and_place_v2 \
  --device cuda:0 \
  --port /dev/ttyACM0 \
  --camera head=0 \
  --camera wrist=2 \
  --fps 30 \
  --inference-mode sync \
  --temporal-ensemble
```

### 运行时行为

- 异步模式下，动作队列为空时会走 `fallback_mode="hold"`，用当前姿态保持。
- 每秒会打印一次状态，包括 `tick`、`source`、`queue`、`latency`、`fallbacks`、`errors`。
- 如果传了 `--debug-actions`，还会打印当前关节、动作、差值和夹爪目标。
- 退出时会先尝试 `hold_current()`，再停止 runtime / SDK，最后断开机器人。

### 直接接手时最容易改的点

- 相机映射：`--camera ROLE=SOURCE`，例如 `head=0`、`wrist=/dev/video2`。
- 串口：`--port`。
- 推理模式：`async` / `sync`。
- 夹爪或动作尺度：如果模型输出和机器人不一致，先核对 6 / 7 维约定。

## 3) `examples/validate_dataset_async_inference.py`

### 这个文件做什么

这是异步 runtime 的数据集验证脚本。它会把 LeRobot 数据集回放一遍，然后把 runtime 实际执行的动作和数据集标注动作做对齐比较，最后输出图、CSV 和汇总结果。

它的主要用途是：

- 验证模型是否能正常加载。
- 验证异步队列、补帧、聚合和 fallback 是否正常。
- 检查动作曲线是否和数据集标注一致。
- 给调参提供可量化指标。

### 输入支持

- 模型来源：本地目录或 Hugging Face repo id。
- 数据集来源：本地 LeRobot 数据集根目录或 Hugging Face dataset repo id。
- episode：单个 episode 或 `--all-episodes`。
- `--instruction`：可选显式指令。

### 播放模式

脚本支持 `--playback-mode offline|fast|realtime`。

**以当前代码为准，默认值是 `offline`。**

- `offline`：走同步 chunk 验证路径，直接用 `predict_action_chunk()` 做对比。
- `fast`：走异步 runtime 路径，但不额外按 dataset FPS 睡眠。
- `realtime`：走异步 runtime 路径，并按 dataset FPS 节奏回放。

如果你要看真实的异步队列行为，通常优先用 `fast` 或 `realtime`；如果你只想先排查预处理和输出尺度，先看 `offline`。

### 输出位置

默认输出目录仍沿用：

```text
outputs/validate_dataset_inference/<timestamp>_<model>_<dataset>_async_<scope>/
```

里面通常会有：

```text
plots/episode_000.png
csv/episode_000.csv
summary.csv
summary.json
```

### 直接可用的命令

最小命令：

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0
```

如果你想显式验证异步队列：

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --playback-mode realtime
```

### 最常用的参数

- `--chunk-size-threshold`：队列补帧阈值。
- `--action-chunk-size`：覆盖 checkpoint 里的 chunk size。
- `--n-action-steps`：覆盖 checkpoint 里的动作步数。
- `--aggregate-fn`：重叠 chunk 的聚合策略。
- `--temporal-ensemble` / `--temporal-ensemble-coeff`：ACT 的时间集成。
- `--enable-rtc`：SmolVLA / PI0 / PI0.5 的 RTC。
- `--disable-gripper-clamping`：做曲线对比时关闭夹爪限速影响。
- `--max-frames`：只跑前几帧，适合快速 debug。
- `--debug-threads` / `--force-exit`：处理第三方库残留线程导致的退出问题。

### 运行时你应该盯的日志

每隔一段，脚本会打印类似信息：

```text
frame 0050/0180 dataset_idx=00123 target_idx=00124 action_ts=49 frame_idx=0049 source=queue queue=12 fallbacks=0 dropped_obs=0 infer_ms=24.8 mae=0.031245
```

重点看这些字段：

- `source=queue`：当前动作来自异步队列。
- `source=fallback_*`：说明队列曾经空过。
- `queue=...`：动作队列剩余长度。
- `fallbacks=...`：fallback 累计次数。
- `dropped_obs=...`：被丢弃的观测数。
- `mae=...`：当前帧的动作误差。

## 建议的接手顺序

如果你是新接手者，建议按这个顺序来：

1. 先跑 `examples/validate_dataset_async_inference.py`，确认模型、数据集和异步 runtime 都能正常工作。
2. 如果要上真机，先跑 `examples/alicia_m_async_runtime.py` 的 `sync` 模式，再切 `async`。
3. 如果你不是 Alicia-M，而是别的硬件平台，直接拿 `examples/async_runtime_loop.py` 做模板改造。

## 修改时要一起检查的地方

如果你改了下面任何一项，最好同时检查这三个文件里的对应逻辑：

- 观测构造方式。
- action / state 的维度和单位。
- 夹爪尺度和 clamping 规则。
- 异步 runtime 的 fallback 策略。
- 现实控制频率 `fps` / `control_fps`。
- 结果输出目录和报告格式。

另外，`examples/validate_dataset_async_inference.py` 复用了 `examples/validate_dataset_inference.py` 里的很多数据集处理、绘图和汇总 helper；如果你改了数据集预处理或曲线对齐逻辑，不要只看异步脚本。

## 常见问题

### 1. `Alicia-M-SDK` 导入失败

- 先确认已经安装。
- 再检查 `ALICIA_M_SDK_PATH`。
- 如果源码仓库在旁边，可以把它放到 `../Alicia-M-SDK`。

### 2. 异步 runtime 启动超时

- 检查相机是否真的打开。
- 检查机器人状态是否已经能读到。
- 适当调大 `--startup-timeout`。
- 先用 `--max-frames` 跑短回放确认流程本身没问题。

### 3. 验证脚本打印完 Summary 但进程不退出

- 先用 `--debug-threads` 看是不是第三方库残留了非 daemon 线程。
- 如果报告已经写完，只想让命令尽快返回，用 `--force-exit`。

### 4. 验证结果看起来整体错位

- 先确认你用的是不是 `offline` 模式。
- 如果是异步模式，重点检查 `action_timestep` 和 target 对齐。
- 再看 `queue`、`fallbacks` 和 `dropped_obs`。

### 5. 真机动作和模型输出不一致

- 先确认动作是 6 维还是 7 维。
- 先确认关节单位是 degree 还是 rad。
- 再确认夹爪值是不是被 clamp 到预期范围。

## 参考入口

- `README.md` 里有这三个脚本的快速示例。
- `examples/validate_dataset_async_inference.md` 记录了异步数据集验证的更详细用法。
- `examples/validate_dataset_inference.md` 记录了同步 / raw 验证路径，很多数据集预处理 helper 和异步脚本共用。
- `docs/async_inference_plan.md` 记录了整个异步 runtime 的设计背景。
