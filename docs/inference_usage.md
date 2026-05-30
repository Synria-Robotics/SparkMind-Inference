# SparkMind Inference Usage

本文档说明 SparkMind 训练产物如何接入 `sparkmind-inference`，以及 ACT / PI0 / PI0.5 在控制环里的推荐用法。

## 基本约定

- Python import 包名是 `sparkmind_inference`。
- 图像输入统一使用 OpenCV BGR `numpy.ndarray`，形状 `(H, W, 3)`。
- `state` 是一维 `numpy.ndarray`，维度必须等于 checkpoint 的 `observation.state`。
- `predict_action_chunk()` 用于离线数值验证或上层自管 action queue。
- `predict_action()` 用于普通同步控制环，内部按 LeRobot v0.5.1 FIFO 语义选择一个动作。
- 每个 episode 开始前调用 `sdk.reset_policy(model_type)`，清空 FIFO、ACT temporal ensemble 和 RTC leftover。

## SparkMind Checkpoint 目录

SparkMind/LeRobot 风格 checkpoint 通常长这样：

```text
checkpoints/050000/
  pretrained_model/
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json
    policy_*_processor.safetensors
  training_state/
    optimizer_state.safetensors
    optimizer_param_groups.json
    rng_state.safetensors
    training_step.json
```

Inference SDK 使用 `pretrained_model/` 推理产物。可以直接传 step 父目录，SDK 会自动解析到 `pretrained_model/`；也可以直接传 `pretrained_model/`。

不要把 `training_state/` 当作推理 checkpoint。它只用于继续训练，里面是 optimizer、RNG 和 step 状态。

## ACT

### 默认 Chunk Queue

普通 ACT / ALOHA checkpoint 常见配置是：

```text
chunk_size = 100
n_action_steps = 100
```

不开 temporal ensemble 时，`predict_action()` 的行为是：

```text
队列空 -> 当前 observation forward 一次，预测 100 步
取前 100 步放入 FIFO
每个 control tick pop 一个 action
队列空后再用新的 observation forward
```

示例：

```python
from sparkmind_inference import InferenceSDK, SmoothingConfig

ckpt = "/path/to/checkpoints/050000"

sdk = InferenceSDK(
    device="cuda:0",
    smoothing_config=SmoothingConfig(enable_gripper_clamping=False),
)
metadata = sdk.load_policy("act", ckpt)

print(metadata.required_cameras)
print(metadata.state_dim, metadata.action_dim)
print(metadata.chunk_size, metadata.n_action_steps)

while running:
    obs = read_observation()
    action = sdk.predict_action(
        "act",
        images={"top": obs.top_bgr},
        state=obs.state_14d,
    )
    send_action(action)
```

如果只想看当前 observation 的完整 chunk：

```python
chunk = sdk.predict_action_chunk(
    "act",
    images={"top": top_bgr},
    state=state_14d,
)
print(chunk.shape)  # e.g. (100, 14)
```

### Temporal Ensemble

ACT temporal ensemble 的语义是：

```text
每个 control tick 都重新预测完整 action chunk
把多次预测中重叠到当前时刻的 action 做在线加权平均
只执行当前 1 步 action
```

因此启用 temporal ensemble 时，`n_action_steps` 必须设为 `1`。LeRobot v0.5.1 也是这个要求。

推荐配置：

```python
from sparkmind_inference import InferenceSDK, SmoothingConfig

sdk = InferenceSDK(
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        enable_temporal_ensemble=True,
        temporal_ensemble_coeff=0.01,
        n_action_steps=1,
        enable_gripper_clamping=False,
    ),
)
sdk.load_policy("act", "/path/to/checkpoints/050000")

for episode in episodes:
    sdk.reset_policy("act")
    for obs in episode:
        action = sdk.predict_action(
            "act",
            images={"top": obs.top_bgr},
            state=obs.state_14d,
        )
        send_action(action)
```

注意：

- temporal ensemble 用 `predict_action()`，不要用 `predict_action_chunk()` 执行控制。
- `temporal_ensemble_coeff=0.01` 是 ACT 常用默认值，已和 LeRobot v0.5.1 对齐验证。
- ALOHA ACT 是 14 维 state/action，不会套用 7 维 Alicia-M 夹爪缩放。

## PI0 / PI0.5 非 RTC

非 RTC 模式是标准 LeRobot FIFO chunk queue：

```text
队列空 -> 当前 observation 推理一个 action chunk
取前 n_action_steps 个 action 放入 FIFO
每个 control tick pop 一个 action
队列空后再重新 forward
```

示例：

```python
from sparkmind_inference import InferenceSDK, SmoothingConfig

sdk = InferenceSDK(
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        enable_rtc=False,
        enable_gripper_clamping=False,
    ),
)

sdk.load_policy(
    "pi05",  # 或 "pi0"
    "/path/to/pi05_checkpoint",
    instruction="put the white mug on the left plate",
)

while running:
    obs = read_observation()
    action = sdk.predict_action(
        "pi05",
        images={
            "image": obs.image_bgr,
            "image2": obs.image2_bgr,
        },
        state=obs.state,
    )
    send_action(action)
```

常见 PI checkpoint 是：

```text
chunk_size = 50
n_action_steps = 50
```

如果想更频繁重规划，可以在推理时覆盖 `n_action_steps`：

```python
sdk = InferenceSDK(
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        enable_rtc=False,
        n_action_steps=10,
    ),
)
```

这会让模型仍然预测完整 `chunk_size`，但 SDK 只执行前 10 步，然后重新根据新 observation 推理。

## PI0 / PI0.5 RTC

RTC 只建议配合 `predict_action_chunk()` 使用。不要在 RTC 模式下调用 `predict_action()` / `engine.step()`；SDK 会直接报错。

原因是 LeRobot 官方也不支持 `select_action()` + RTC。RTC 需要上层控制器管理 chunk 执行队列、inference delay 和 leftover chunk。

示例：

```python
from collections import deque

from sparkmind_inference import InferenceSDK, SmoothingConfig

execution_horizon = 10
action_queue = deque()

sdk = InferenceSDK(
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        enable_rtc=True,
        rtc_prefix_attention_schedule="LINEAR",
        rtc_execution_horizon=execution_horizon,
        rtc_inference_delay_steps=0,
        rtc_max_guidance_weight=10.0,
        enable_gripper_clamping=False,
    ),
)

sdk.load_policy(
    "pi05",  # 或 "pi0"
    "/path/to/pi05_checkpoint",
    instruction="put the white mug on the left plate",
)

for episode in episodes:
    sdk.reset_policy("pi05")
    action_queue.clear()

    while running:
        obs = read_observation()

        if not action_queue:
            chunk = sdk.predict_action_chunk(
                "pi05",
                images={
                    "image": obs.image_bgr,
                    "image2": obs.image2_bgr,
                },
                state=obs.state,
            )
            for action in chunk[:execution_horizon]:
                action_queue.append(action)

        send_action(action_queue.popleft())
```

参数含义：

- `rtc_execution_horizon`：上层这次会执行多少步 action。
- `rtc_inference_delay_steps`：静态推理延迟补偿，单位是 control tick；如果测到推理慢 2 个 tick，可以设为 `2`。
- `rtc_prefix_attention_schedule`：RTC 前缀注意力权重，常用 `LINEAR`。
- `sdk.reset_policy("pi05")`：清空 RTC leftover，必须在新 episode 开始调用。

## Gripper Clamping

`enable_gripper_clamping` 是部署保护项，用来限制 7 维机器人 action 最后一维夹爪目标的变化速度，例如 Alicia-M 的 `[joint_1..joint_6, gripper_0_to_1000]`。

做 SDK vs LeRobot 正确性验证、ALOHA ACT 14D 推理、PI/SmolVLA 离线对齐时建议关闭：

```python
SmoothingConfig(enable_gripper_clamping=False)
```

开启后 SDK 会改写模型原始 action 输出，因此不要用开启 clamping 的结果做数值正确性判断。

## 验证命令

ACT temporal ensemble 对齐：

```bash
python examples/compare_lerobot_official_inference.py \
  --model /path/to/act_checkpoint \
  --model-type act \
  --dataset lerobot/aloha_sim_transfer_cube_human \
  --episode 0 \
  --max-frames 5 \
  --device cuda:0 \
  --dataset-gripper-scale raw \
  --temporal-ensemble \
  --temporal-ensemble-coeff 0.01
```

PI0 / PI0.5 raw chunk 对齐：

```bash
python examples/compare_lerobot_official_inference.py \
  --model lerobot/pi05_libero_base \
  --model-type pi05 \
  --dataset lerobot/libero \
  --episode 0 \
  --max-frames 3 \
  --device cuda:0
```

PI0 / PI0.5 RTC chunk 对齐：

```bash
python examples/compare_lerobot_official_inference.py \
  --model lerobot/pi0_libero_base \
  --model-type pi0 \
  --dataset lerobot/libero \
  --episode 0 \
  --max-frames 5 \
  --device cuda:0 \
  --enable-rtc \
  --preserve-policy-state \
  --rtc-execution-horizon 10 \
  --rtc-inference-delay-steps 0
```
