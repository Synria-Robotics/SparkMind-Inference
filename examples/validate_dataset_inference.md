# `validate_dataset_inference.py` 使用说明

## 作用

`examples/validate_dataset_inference.py` 用来在 LeRobot / LIBERO 风格数据集上回放 observation，并把 SDK 预测动作和数据集标注动作逐帧对比。

脚本会输出：

- 每个 episode 的动作曲线图
- 每个 episode 的逐帧 CSV
- 汇总 `summary.json` 和 `summary.csv`

它主要用于确认模型加载、相机映射、图像预处理、state/action 归一化、夹爪尺度和 checkpoint 配置是否正确。

## 前提条件

建议先进入仓库根目录并激活环境：

```bash
cd /home/synria/demo/Inference-SDK
source .venv/bin/activate
```

安装验证脚本依赖：

```bash
uv pip install -e ".[all,examples]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果使用本地 `SparkMind` checkout，推荐放在仓库内的 `third_party/SparkMind` 并安装：

```bash
mkdir -p third_party
git clone https://github.com/Synria-Robotics/SparkMind.git -b dev_ch_v0.1 third_party/SparkMind
uv pip install -e third_party/SparkMind -i https://pypi.tuna.tsinghua.edu.cn/simple
```

预训练模型和数据集可以从 Hugging Face 下载。`--model` 和 `--dataset` 都可以传本地路径或 Hugging Face repo id；传 repo id 时脚本会自动下载并缓存到 `.cache/huggingface/`。

如果校园网访问 Hugging Face 需要代理，先设置：

```bash
export HTTP_PROXY=http://proxy.cse.cuhk.edu.hk:8000
export HTTPS_PROXY=http://proxy.cse.cuhk.edu.hk:8000
export ALL_PROXY=http://proxy.cse.cuhk.edu.hk:8000
```

也可以先手动下载到本地：

```bash
hf download <model_repo_id> --repo-type model --local-dir models/<model_name>
hf download <dataset_repo_id> --repo-type dataset --local-dir data/lerobot/<dataset_name>
```

## 验证模式

`validate_dataset_inference.py` 有两种同步验证模式：

- `raw`：每帧调用 `engine.predict_chunk()`，比较 `chunk[0]` 和 dataset action。默认模式，最适合验证 inference engine 本身的数值正确性。
- `step`：每帧调用 `engine.step()`，验证同步控制环实际会执行的动作。适合验证 ACT temporal ensemble 或 SmolVLA/PI0/PI0.5 RTC。

默认 `--execution-mode auto`：开启 `--temporal-ensemble` 时自动走 `step`，其他情况走 `raw`。

## 常用示例

验证一个 ACT 模型：

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --device cuda:0
```

验证 ACT 同步时间集成：

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --temporal-ensemble
```

显式验证同步 `step()` 控制环：

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --execution-mode step
```

验证 PI0 / PI0.5 / SmolVLA：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/pi0_checkpoint \
  --model-type pi0 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick up the object"
```

验证开启 RTC 的 PI0 / PI0.5 / SmolVLA：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/pi05_checkpoint \
  --model-type pi05 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick and place" \
  --execution-mode step \
  --enable-rtc \
  --rtc-prefix-attention-schedule LINEAR \
  --rtc-execution-horizon 10 \
  --rtc-inference-delay-steps 0
```

只跑前 50 帧做快速调试：

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --episode 0 \
  --max-frames 50
```

## 常用参数

- `--model`：模型目录，或 Hugging Face 模型 repo id。
- `--model-type`：算法类型，支持 `act`、`pi0`、`pi05`、`smolvla`；不传时从模型配置推断。
- `--dataset`：LeRobot 数据集根目录，或 Hugging Face dataset repo id。
- `--episode` / `--all-episodes`：指定单个 episode 或验证全部 episode。
- `--device`：推理设备，例如 `cuda:0` 或 `cpu`。
- `--instruction`：语言指令，主要用于 `pi0` / `pi05` / `smolvla`；不传时优先使用数据集样本里的 `task` 字段。
- `--execution-mode`：`auto` / `raw` / `step`。
- `--temporal-ensemble`：开启 ACT 同步时间集成，只支持 `act` + `step`/`auto`。
- `--temporal-ensemble-coeff`：ACT 时间集成系数，默认 `0.01`，必须和 `--temporal-ensemble` 一起使用。
- `--action-chunk-size`：覆盖 checkpoint `chunk_size`，即模型一次 forward 的动作 horizon。
- `--n-action-steps`：覆盖 checkpoint `n_action_steps`，即每次推理实际返回的动作数。
- `--enable-rtc`：为 `smolvla` / `pi0` / `pi05` 开启 RTC。
- `--dataset-gripper-scale`：夹爪值缩放模式，通常保留默认 `auto`。
- `--video-backend`：LeRobotDataset 使用的视频后端，默认 `pyav`。
- `--max-frames`：只处理前 N 帧，便于调试。

## 输出结果

如果不指定 `--output-dir`，结果默认写到：

```bash
outputs/validate_dataset_inference/<时间戳>_<模型名>_<数据集名>_<scope>/
```

目录结构：

```text
outputs/validate_dataset_inference/20260422_170000_model_dataset_episode_000/
├── plots/
│   └── episode_000.png
├── csv/
│   └── episode_000.csv
├── summary.csv
└── summary.json
```

## 建议排查顺序

先跑 `raw` 模式。如果 `raw` 也不对，优先检查 checkpoint、`config.json`、processor stats、模型类型、instruction、相机 key、RGB/BGR 和夹爪尺度。

如果 `raw` 正常但 `step` 不对，再检查 `control_fps`、`n_action_steps`、ACT temporal ensemble 或 RTC 参数。

## 下一步：仿真闭环

离线数值验证通过后，用 LeRobot official env wrapper 做任务级 smoke test，并同时记录 official policy action 与 SDK action：

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

该脚本会输出 success、return、逐步 action diff，并保存 official `env.render()` 同视角视频。SmolVLA 的 LeRobot official 闭环使用 FIFO chunk queue；仿真正确性不要用 wall-clock timestamp queue 判断。
