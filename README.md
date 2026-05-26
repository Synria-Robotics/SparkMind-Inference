# Inference SDK

`inference-sdk` 是一个独立的 Python SDK，用于 ACT、SmolVLA、PI0 等 policy 模型推理。PI0.5 入口保留为 experimental。

它只关注一件事：

> 输入 observation，输出 action / action chunk。

硬件驱动、相机采集、Web API、会话编排、任务状态机等业务逻辑应放在上层业务应用中。

## 安装

推荐使用 `uv` 管理隔离环境：

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
```

SDK 的 ACT / SmolVLA / PI0 / PI0.5 engine 依赖 `SparkMind` 模型实现。推荐先安装本地 `SparkMind`，再安装 Inference-SDK：

```bash
mkdir -p third_party
git clone https://github.com/Synria-Robotics/SparkMind.git -b dev_ch_v0.1 third_party/SparkMind
uv pip install -e "third_party/SparkMind[pi,libero]" -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -e ".[all,examples]" -c constraints/validated.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果 SparkMind checkout 在其他位置，请先安装该路径，并设置 `INFERENCE_SDK_SPARKMIND_PATH` 指向它。`pyproject.toml` 会声明 `sparkmind>=1.0.0`，但不会把 SDK 绑死到某个远端 SparkMind commit；这样本地迁移分支可以作为唯一实现来源。

可选依赖：

```bash
uv pip install -e ".[act]" -c constraints/validated.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -e ".[examples]" -c constraints/validated.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -e ".[vla]" -c constraints/validated.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -e ".[all]" -c constraints/validated.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果需要运行 dataset 验证和绘图示例，推荐直接安装：

```bash
uv pip install -e ".[all,examples]" -c constraints/validated.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`pyproject.toml` 使用经过验证的版本范围；`constraints/validated.txt` 固定当前验证环境的顶层依赖版本。当前验证主线是 Python 3.13、外部 LeRobot v0.5.1、SparkMind 1.0.0、transformers 5.3.0。

## Hugging Face 下载

预训练好的模型和 LeRobot / LIBERO 数据集可以直接从 Hugging Face 下载。`examples/validate_dataset_inference.py` 的 `--model` 和 `--dataset` 都支持传本地路径或 Hugging Face repo id；传 repo id 时脚本会自动调用 `huggingface_hub.snapshot_download()`，并缓存到仓库内 `.cache/huggingface/`。

如果在校园网环境需要代理，可以先设置：

```bash
export HTTP_PROXY=http://proxy.cse.cuhk.edu.hk:8000
export HTTPS_PROXY=http://proxy.cse.cuhk.edu.hk:8000
export ALL_PROXY=http://proxy.cse.cuhk.edu.hk:8000
```

也可以先手动下载到本地再运行验证：

```bash
hf download <model_repo_id> --repo-type model --local-dir models/<model_name>
hf download <dataset_repo_id> --repo-type dataset --local-dir data/lerobot/<dataset_name>
```

## Public API Contract

- 主入口从顶层导入：`InferenceSDK`、`Observation`、`SmoothingConfig`、`create_engine`、`predict_action_chunk`、`predict_action` 和 SDK 自定义异常。
- `images` 使用相机角色名作为 key，例如 `head`、`wrist`；可用角色以 `metadata.required_cameras` 为准。
- 每张图像应是 BGR 格式的 `numpy.ndarray`，形状为 `(H, W, 3)`。
- `state` 应是一维 `numpy.ndarray`，维度需要和模型 `observation.state` 一致。
- ACT、SmolVLA、PI0 作为 stable 推理路径；PI0.5 可通过 `algorithm_type="pi05"` 使用，但当前标记为 experimental。
- SDK 对外按 robot-space 处理夹爪，输入/输出最后一维夹爪通常是 `[0, 1000]`。
- `predict_action_chunk()` 和 `predict_action()` 都在调用线程同步执行，不启动后台推理线程。
- `load_policy()`、observation 校验和推理错误会抛出 SDK 自定义异常，方便上位机按错误类型处理。
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

真机控制可以使用同步单步 `predict_action()`。这会在调用线程内执行 `engine.step()` 并直接返回当前 tick 的一个 action；ACT 可配合同步时间集成提升控制输出稳定性：

```python
from inference_sdk import InferenceSDK, Observation, SmoothingConfig

with InferenceSDK(
    device="cuda:0",
    smoothing_config=SmoothingConfig(
        control_fps=30.0,
        enable_temporal_ensemble=True,
        temporal_ensemble_coeff=0.01,
    ),
) as sdk:
    metadata = sdk.load_policy(
        algorithm_type="act",
        checkpoint_dir="/path/to/checkpoint",
    )

    while True:
        observation = Observation(images=read_camera_images(), state=read_robot_state())
        action = sdk.predict_action("act", observation)
        send_robot_action(action)
```

PI0.5 可使用 `algorithm_type="pi05"`，`"pi0.5"` / `"pi0_5"` / `"pi0-5"` 也会自动归一到 `pi05`。该路径当前为 experimental，建议发布口径中优先使用 ACT、SmolVLA 和 PI0。

如果只需要执行一次推理，也可以使用一次性 API：

```python
from inference_sdk import predict_action, predict_action_chunk

action_chunk = predict_action_chunk(
    algorithm_type="act",
    checkpoint_dir="/path/to/checkpoint",
    images=images,
    state=robot_state,
)

action = predict_action(
    algorithm_type="act",
    checkpoint_dir="/path/to/checkpoint",
    images=images,
    state=robot_state,
)
```

## Advanced Engine API

底层 engine API 仍然保留，适合需要直接控制 `load()`、`reset()`、`predict_chunk()`、`step()` 的场景。对外集成优先使用高层 `InferenceSDK`；engine、queue、trace 等内部类型不再从顶层包导出。

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

SmolVLA、PI0 可以开启 RTC；PI0.5 的 RTC 路径保留为 experimental。`rtc_inference_delay_steps` 是静态延迟步数；如果你的控制环能测出实际推理延迟，可以按 tick 数传入：

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

说明：ACT 时间集成当前用于同步 `engine.step()` / `select_action()`；SmolVLA、PI0 的 RTC 也在同步推理路径中生效。

## 同步参数

- `control_fps`：控制频率；同步 `step()` 会据此计算动作队列时间步。
- `action_chunk_size`：可选，覆盖 checkpoint 里的 `chunk_size`，表示模型一次 forward 的动作 horizon。
- `n_action_steps`：可选，覆盖 checkpoint 里的 `n_action_steps`，表示每次推理实际返回并进入动作队列的动作数，必须 `<= action_chunk_size/chunk_size`。
- `aggregate_fn_name`：同步 `step()` 内部动作队列出现重叠 timestep 时的融合方式，支持 `latest_only`、`weighted_average`、`average`、`conservative`。
- `fallback_mode`：同步动作队列为空时的行为；`hold` 使用当前 robot state，`repeat` 重复上一条动作。
- `enable_gripper_clamping`：是否对夹爪动作做速度限制，真机可开启，离线曲线对比时可关闭。
- `enable_temporal_ensemble`：为 ACT 的同步 `step()` 开启在线时间集成。
- `enable_rtc`：为 SmolVLA / PI0 开启 RTC；PI0.5 路径为 experimental。
- `rtc_prefix_attention_schedule`：RTC 前缀注意力权重，支持 `ZEROS`、`ONES`、`LINEAR`、`EXP`。
- `rtc_execution_horizon` / `rtc_inference_delay_steps`：RTC 执行窗口和静态推理延迟步数。

## Alicia-M 真机控制

`examples/alicia_m_sync_runtime.py` 直接调用 `Alicia-M-SDK`，使用同步 `InferenceSDK.predict_action()`：

- 从 `Alicia-M-SDK` 读取 `[6 关节 rad, gripper 0~1000]`，送入 policy 前把 6 个关节转换成 degree。
- 默认按绝对关节/夹爪目标发布，`action[:6]` 是 6 个关节 degree 目标，脚本用 `joint_format="deg"` 交给 `Alicia-M-SDK`；7 维 action 的 `action[6]` 是夹爪 `[0, 1000]`。

如果 `Alicia-M-SDK` 没安装成包，示例会尝试加载同级目录 `../Alicia-M-SDK`；也可以显式指定：

```bash
export ALICIA_M_SDK_PATH=/path/to/Alicia-M-SDK
```

OpenCV 相机示例：

```bash
python examples/alicia_m_sync_runtime.py \
  --model-type act \
  --checkpoint-dir /path/to/act_checkpoint \
  --device cuda:0 \
  --port /dev/ttyACM0 \
  --camera head=0 \
  --camera wrist=2 \
  --fps 30
```

ACT 可以开启同步时间集成：

```bash
python examples/alicia_m_sync_runtime.py \
  --model-type act \
  --checkpoint-dir /path/to/act_checkpoint \
  --device cuda:0 \
  --port /dev/ttyACM0 \
  --camera head=0 \
  --camera wrist=2 \
  --fps 30 \
  --temporal-ensemble
```

`act` 可加 `--temporal-ensemble`，`smolvla` / `pi0` / `pi05` 可加 `--enable-rtc`。

## Dataset 验证

离线逐帧验证是当前正确性验证主线，用于确认模型预处理、权重加载和输出尺度是否正确。默认 `raw` 模式每帧调用 `predict_chunk()`，比较 `chunk[0]` 和 dataset action：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/act_checkpoint \
  --model-type act \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --execution-mode raw
```

同步 `step` 模式用于验证控制环真实会执行到的动作，适合 ACT temporal ensemble 或 RTC：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/act_checkpoint \
  --model-type act \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --execution-mode step \
  --temporal-ensemble
```

PI0 / PI0.5 示例：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/pi05_checkpoint \
  --model-type pi05 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick and place" \
  --enable-rtc \
  --rtc-execution-horizon 10
```

如果只是做曲线对比，优先使用 `raw` 模式排除同步动作队列影响：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/pi05_checkpoint \
  --model-type pi05 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick and place" \
  --execution-mode raw
```

验证结果会写入：

```text
outputs/validate_dataset_inference/<timestamp>_<model>_<dataset>_<scope>/
├── plots/
├── csv/
├── summary.csv
└── summary.json
```

更多说明见 `examples/validate_dataset_inference.md`。

## LIBERO 仿真闭环验证

SmolVLA 的 LIBERO 闭环验证应优先使用 LeRobot official env wrapper，并在同一 observation 上同时计算 official policy action 和 SDK action。这样可以排除裸 `OffScreenRenderEnv` 的 reset、camera flip、settle step 和 action queue 差异。

```bash
python examples/compare_libero_sim_actions.py \
  --model lerobot/smolvla_libero \
  --benchmark libero_spatial \
  --task-id 0 \
  --max-steps 280 \
  --device cuda:0 \
  --execute sdk \
  --sdk-selection fifo \
  --save-video
```

该脚本会保存 `action_comparison.csv`、`summary.json` 和 `rollout_render.mp4`。SmolVLA official eval 使用 FIFO chunk queue；不要用 wall-clock timestamp queue 判断 LIBERO 仿真正确性。

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
- 如果 raw 正常但 `step` 不对，再检查 `n_action_steps`、`control_fps`、ACT 时间集成或 RTC 配置。
- PI0 / PI0.5 的权重加载现在会严格校验 missing/unexpected keys；如果 checkpoint 没有干净加载，SDK 会直接报错，避免误用随机初始化权重。
- LeRobot 官方 PI checkpoint 会尽量保持和官方 processor 一致的 state/action 边界语义；legacy 导出格式继续使用 SDK stats 归一化。

### 离线环境加载 PI0 / PI0.5 tokenizer 失败

请先把 tokenizer 下载到本地，并通过环境变量指定：

```bash
export PI0_TOKENIZER_PATH=/path/to/local/paligemma-tokenizer
# PI0.5 也可使用专用环境变量
export PI05_TOKENIZER_PATH=/path/to/local/paligemma-tokenizer
```
