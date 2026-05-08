# Inference SDK

`inference-sdk` 是一个独立的 Python SDK，用于 ACT、SmolVLA、PI0、PI0.5 等 policy 模型推理。

它只关注一件事：

> 输入 observation，输出 action / action chunk。

硬件驱动、相机采集、Web API、会话编排、任务状态机等业务逻辑应放在上层业务应用中。

## 安装

推荐使用 `uv` 管理隔离环境：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install -e ../SparkMind
```

可选依赖：

```bash
uv pip install -e .[act]
uv pip install -e .[examples]
uv pip install -e .[vla]
uv pip install -e .[all]
```

如果需要运行 dataset 验证和绘图示例，推荐直接安装：

```bash
uv pip install -e ".[all,examples]"
```

## 数据约定

- `images` 使用相机角色名作为 key，例如 `head`、`wrist`；可用角色以 `metadata.required_cameras` 为准。
- 每张图像应是 BGR 格式的 `numpy.ndarray`，形状为 `(H, W, 3)`。
- `state` 应是一维 `numpy.ndarray`，维度需要和模型 `observation.state` 一致。
- 当前 ACT、SmolVLA、PI0、PI0.5 engine 默认按 robot-space 处理夹爪，输入/输出最后一维夹爪通常是 `[0, 1000]`。
- LeRobot dataset 里的夹爪如果是归一化 `[0, 1]`，验证脚本会通过 `--dataset-gripper-scale auto` 自动适配。

## 同步推理

推荐给业务应用使用高层 `InferenceSDK` API。它负责加载 policy、校验 observation，并输出一个动作 chunk。

```python
from inference_sdk import InferenceSDK, Observation

with InferenceSDK(device="cuda:0") as sdk:
    metadata = sdk.load_policy(
        algorithm_type="pi0",
        checkpoint_dir="/path/to/checkpoint",
        instruction="Pick up the object.",
    )

    observation = Observation(
        images={
            "head": head_bgr,
            "wrist": wrist_bgr,
        },
        state=robot_state,
    )

    action_chunk = sdk.predict_action_chunk("pi0", observation)
    print(action_chunk.shape)  # (metadata.n_action_steps, metadata.action_dim)
```

PI0.5 可使用 `algorithm_type="pi05"`，`"pi0.5"` / `"pi0_5"` / `"pi0-5"` 也会自动归一到 `pi05`。

如果只需要执行一次推理，也可以使用一次性 API：

```python
from inference_sdk import predict_action_chunk

action_chunk = predict_action_chunk(
    algorithm_type="act",
    checkpoint_dir="/path/to/checkpoint",
    images=images,
    state=robot_state,
)
```

## 底层 Engine API

底层 engine API 仍然保留，适合需要直接控制 `load()`、`reset()`、`predict_chunk()`、`step()` 的场景。

```python
from inference_sdk import SmoothingConfig, create_engine

engine = create_engine(
    model_type="act",
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        control_fps=30.0,
        aggregate_fn_name="latest_only",
    ),
)

ok, error = engine.load("/path/to/checkpoint")
if not ok:
    raise RuntimeError(error)

try:
    engine.reset()
    action_chunk = engine.predict_chunk(images, state)
    action = engine.step(images, state)
finally:
    engine.unload()
```

说明：

- `predict_chunk()`：直接返回当前 observation 对应的完整动作 chunk，适合离线验证和曲线对比。
- `step()` / `select_action()`：走 engine 内部动作队列，每次返回一个控制 tick 的动作。

### 可选推理增强

ACT 可以在同步 `engine.step()` 路径开启 LeRobot 风格时间集成。开启后每个控制 tick 都会重新预测一个 action chunk，并在线融合重叠时间步的动作：

```python
engine = create_engine(
    model_type="act",
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        enable_temporal_ensemble=True,
        temporal_ensemble_coeff=0.01,
    ),
)
```

SmolVLA、PI0、PI0.5 可以开启 RTC。`rtc_inference_delay_steps` 是静态延迟步数；如果你的控制环能测出实际推理延迟，可以按 tick 数传入：

```python
engine = create_engine(
    model_type="pi05",
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        enable_rtc=True,
        rtc_prefix_attention_schedule="LINEAR",
        rtc_execution_horizon=10,
        rtc_inference_delay_steps=0,
    ),
)
```

说明：ACT 时间集成当前用于同步 `engine.step()` / `select_action()`；进程内异步 runtime 仍使用 action chunk 队列和 `aggregate_fn_name` 处理重叠 chunk。

## 进程内异步推理

如果需要将“动作执行”和“模型推理”解耦，可以使用 `AsyncInferenceRuntime`。它不启动 gRPC server/client，而是在当前 Python 后端进程内维护：

- 观测队列：默认只保留最新观测，避免处理过期帧。
- 后台推理线程：持续根据最新观测预测后续动作 chunk。
- 动作队列：前台控制循环按控制频率取动作执行。

真机控制建议启用 `warmup + wait_until_ready`，并显式配置 fallback 行为。

```python
import time

from inference_sdk import AsyncInferenceConfig, get_global_async_runtime

fps = 30.0
runtime = get_global_async_runtime()

runtime.load_policy(
    algorithm_type="act",
    checkpoint_dir="/path/to/checkpoint",
    device="cuda:0",
    config=AsyncInferenceConfig(
        control_fps=fps,
        chunk_size_threshold=0.5,
        # action_chunk_size=50,  # 可选：覆盖 checkpoint 里的 chunk_size
        # n_action_steps=10,     # 可选：每次推理实际返回/入队的动作数
        aggregate_fn_name="weighted_average",
        fallback_mode="hold",
    ),
)

runtime.warmup(
    images=read_camera_images(),
    state=read_robot_state(),
)
runtime.start()

if not runtime.wait_until_ready(min_queue_size=1, timeout=5.0):
    runtime.stop()
    raise RuntimeError("Async runtime did not produce an initial action.")

try:
    while True:
        tick_start = time.monotonic()
        result = runtime.step(
            images=read_camera_images(),
            state=read_robot_state(),
        )
        send_robot_action(result.action)

        elapsed = time.monotonic() - tick_start
        time.sleep(max(0.0, 1.0 / fps - elapsed))
finally:
    runtime.stop()
```

`runtime.step()` 返回 `AsyncStepResult`，常用字段包括：

- `action`：当前 tick 应执行的动作。
- `source`：动作来源，例如 `queue`、`fallback_hold`、`fallback_repeat`。
- `queue_size`：当前动作队列剩余动作数。
- `latency_estimate`：推理耗时估计。
- `action_timestep`：该动作对应的队列 timestep，便于离线验证对齐 target。

完整控制循环模板见 `examples/async_runtime_loop.py`，设计方案见 `docs/async_inference_plan.md`。

## 异步参数

- `control_fps`：控制频率；真实机器人执行周期应与该值一致。
- `chunk_size_threshold`：当 `action_queue_size / chunk_size <= threshold` 时提交新观测触发后续推理。
- `action_chunk_size`：可选，覆盖 checkpoint 里的 `chunk_size`，表示模型一次 forward 的动作 horizon。
- `n_action_steps`：可选，覆盖 checkpoint 里的 `n_action_steps`，表示每次推理实际返回并进入动作队列的动作数，必须 `<= action_chunk_size/chunk_size`。
- `aggregate_fn_name`：新旧动作 chunk 重叠 timestep 的融合方式，支持 `latest_only`、`weighted_average`、`average`、`conservative`。
- `enable_temporal_ensemble`：为 ACT 开启异步队列层时间集成。开启后，重叠 timestep 使用 ACT 指数权重融合；通常需要 `n_action_steps > 1` 且队列补帧有重叠才明显生效。
- `fallback_mode`：动作队列为空时的行为；`hold` 使用当前 robot state，`repeat` 重复上一条动作。
- `enable_gripper_clamping`：是否对夹爪动作做速度限制，真机可开启，离线曲线对比时可关闭。
- `enable_rtc`：为 SmolVLA / PI0 / PI0.5 开启 RTC。
- `rtc_prefix_attention_schedule`：RTC 前缀注意力权重，支持 `ZEROS`、`ONES`、`LINEAR`、`EXP`。
- `rtc_execution_horizon` / `rtc_inference_delay_steps`：RTC 执行窗口和静态推理延迟步数。

## Alicia-M 真机控制

`examples/alicia_m_async_runtime.py` 直接调用 `Alicia-M-SDK` 和 `AsyncInferenceRuntime`：

- 从 `Alicia-M-SDK` 读取 `[6 关节 rad, gripper 0~1000]` 作为 policy state。
- 调用 SDK async runtime 得到 action。
- 默认按绝对关节/夹爪目标发布，`action[:6]` 是 6 个关节弧度目标；7 维 action 的 `action[6]` 是夹爪 `[0, 1000]`。
- 异步动作队列为空时使用 `fallback_mode="hold"`，直接保持当前 robot state。

如果 `Alicia-M-SDK` 没安装成包，示例会尝试加载同级目录 `../Alicia-M-SDK`；也可以显式指定：

```bash
export ALICIA_M_SDK_PATH=/path/to/Alicia-M-SDK
```

OpenCV 相机示例：

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

这个真机示例和 `examples/async_runtime_loop.py` 保持同一套最小推理参数，只额外需要 Alicia-M 串口和 OpenCV 相机映射。`act` 可加 `--temporal-ensemble`，`smolvla` / `pi0` / `pi05` 可加 `--enable-rtc`。

## Dataset 验证

同步 / raw 验证用于确认模型预处理、权重加载和输出尺度是否正确：

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --execution-mode raw
```

进程内异步验证用于验证动作队列、观测队列和后台推理调度：

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0
```

异步验证脚本默认使用 `--playback-mode realtime`，会按 dataset FPS 播放，避免大模型在最快速离线循环下拿不到真实控制周期。

PI0 / PI0.5 示例：

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/pi05_checkpoint \
  --model-type pi05 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick and place" \
  --enable-rtc \
  --rtc-execution-horizon 10 \
  --debug-threads
```

如果只是做曲线对比，想排除异步融合和夹爪限速影响，可以使用：

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/pi05_checkpoint \
  --model-type pi05 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick and place" \
  --aggregate-fn latest_only \
  --disable-gripper-clamping
```

验证结果会写入：

```text
outputs/validate_dataset_inference/<timestamp>_<model>_<dataset>_<scope>/
├── plots/
├── csv/
├── summary.csv
└── summary.json
```

更多说明见 `examples/validate_dataset_inference.md` 和 `examples/validate_dataset_async_inference.md`。

## 常见问题

### PI0 / PI0.5 曲线明显不对

建议先用同步 raw 路径确认模型本身：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/pi0_checkpoint \
  --model-type pi0 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick and place" \
  --execution-mode raw
```

- 如果 raw 也不对，优先检查 checkpoint、`config.json`、processor stats、模型类型和 instruction。
- 如果 raw 正常但 async 不对，再检查 `dropped_obs`、`last_infer_ms`、`queue_size`、`fallbacks` 等异步 runtime 日志。
- PI0 / PI0.5 会按 checkpoint 的 `normalization_mapping` 做 state/action 归一化和反归一化；如果权重加载出现大量 missing/unexpected keys，启动日志会给出 warning。

### 程序打印 Summary 后不退出

部分 tokenizer / CUDA / 第三方库可能残留非 daemon 线程。可先定位：

```bash
python examples/validate_dataset_async_inference.py ... --debug-threads
```

如果确认报告已写完，只需要让命令行返回，可以加：

```bash
python examples/validate_dataset_async_inference.py ... --force-exit
```

### 离线环境加载 PI0 / PI0.5 tokenizer 失败

请先把 tokenizer 下载到本地，并通过环境变量指定：

```bash
export PI0_TOKENIZER_PATH=/path/to/local/paligemma-tokenizer
# PI0.5 也可使用专用环境变量
export PI05_TOKENIZER_PATH=/path/to/local/paligemma-tokenizer
```
