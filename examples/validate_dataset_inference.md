# `validate_dataset_inference.py` 使用说明

## 作用

`examples/validate_dataset_inference.py` 用来在离线数据集上跑一次 SDK 推理，并把模型输出和数据集里的标注动作做逐帧对比。

脚本会输出：

- 每个 episode 的动作曲线图
- 每个 episode 的逐帧 CSV
- 汇总 `summary.json` 和 `summary.csv`

适合用来快速检查：

- 模型能否正常加载
- 模型输出维度是否和数据集一致
- 模型在数据集回放上的误差大致有多大
- `predict_chunk()` 和 `step()` 两条执行路径是否符合预期

## 前提条件

建议先进入仓库根目录并激活环境：

```bash
cd /home/synria/demo/Inference-SDK
source .venv/bin/activate
```

如果当前环境还没装完整依赖，至少需要保证脚本相关依赖可用，例如：

```bash
uv pip install -e ".[all,examples]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果你使用本地 `SparkMind` checkout，推荐放在仓库内的 `third_party/SparkMind` 并额外安装：

```bash
mkdir -p third_party
git clone https://github.com/Synria-Robotics/SparkMind.git -b dev_ch_v0.1 third_party/SparkMind
uv pip install -e third_party/SparkMind -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果已经进入 `third_party/SparkMind` 目录，也可以执行 `uv pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple`。

你还需要准备：

- 一个可加载的导出模型目录，或者 Hugging Face 模型 repo id
- 一个 LeRobot 格式的数据集根目录，或者 Hugging Face dataset repo id

## 最基本的用法

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --episode 0
```

这里的 `--model-type` 现在支持直接指定算法类型：

- `act`
- `pi0`
- `pi05`
- `smolvla`

如果不传 `--model-type`，脚本会尝试从模型目录里的配置自动推断。

`validate_dataset_inference.py` 现在有两种验证模式：

- `raw`
  脚本每一帧调用一次 `predict_chunk()`，并把返回 `action chunk` 的第一个动作拿来和数据集标注对比。
- `step`
  脚本每一帧调用一次 `step()`，验证控制环真实会执行到的动作。
  异步推理只在这个模式下生效。

默认是 `--execution-mode auto`：

- 没有请求时间集成和异步推理时，自动走 `raw`
- 传了 `--temporal-ensemble` 或 `--enable-async-inference` 时，自动切到 `step`

如果传了 `--temporal-ensemble`，脚本会开启 SDK 内 ACT engine 的 LeRobot 风格时间集成，并通过 `step()` 验证控制环真实输出。

默认系数是 `0.01`。如果要覆盖它，再额外传：

```bash
--temporal-ensemble-coeff 0.01
```

注意：

- `--temporal-ensemble-coeff` 不能单独使用
- 不传 `--temporal-ensemble` 就代表关闭时间集成
- `--temporal-ensemble` 只支持 ACT
- 如果显式指定 `--execution-mode raw`，就不能再传 `--temporal-ensemble`
- 如果显式指定 `--execution-mode raw`，就不能再传 `--enable-async-inference`
- 当前 example 中，`--temporal-ensemble` 和 `--enable-async-inference` 不能同时开启
- `--enable-rtc` 只支持 `smolvla`、`pi0`、`pi05`，不传就保持关闭

## 常用示例

### 1. 验证一个本地 ACT 模型

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --device cuda:0
```

说明：

- 这条命令默认会走 `raw` 模式
- 它验证的是 `predict_chunk()` 的首个动作

### 1.1 验证带时间集成的 ACT 控制环

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --temporal-ensemble
```

说明：

- 因为传了 `--temporal-ensemble`，默认会自动切到 `step` 模式
- 这时候你看到的曲线会反映 chunk 重叠动作经指数加权后的执行结果

### 1.2 验证 step 控制环

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --execution-mode step
```

说明：

- 这条命令会调用底层 engine 的同步 `step()` 队列路径
- 如需验证进程内异步 runtime，可加 `--enable-async-inference`
- 异步验证脚本会按 runtime 返回的 `action_timestep` 对齐 dataset target，避免队列动作看起来整体慢一帧
- 异步验证脚本默认使用 `--playback-mode realtime`，按 dataset FPS 播放；大模型不要用最快速离线循环判断曲线，否则后台推理线程拿不到真实控制周期

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0
```

### 2. 验证一个 PI0 模型

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/pi0_checkpoint \
  --model-type pi0 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick up the object"
```

说明：

- `PI0`、`PI0.5` 和 `SmolVLA` 可以通过 `--instruction` 显式传入指令。
- 如果不传，脚本会优先使用数据集样本里的 `task` 字段。

### 2.1 验证开启 RTC 的 PI0 / PI0.5 / SmolVLA

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/pi05_checkpoint \
  --model-type pi05 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick and place" \
  --enable-rtc \
  --rtc-prefix-attention-schedule LINEAR \
  --rtc-execution-horizon 10 \
  --rtc-inference-delay-steps 0
```

说明：

- `--enable-rtc` 只支持 `smolvla`、`pi0`、`pi05`
- `--rtc-inference-delay-steps` 是静态控制步延迟，默认 `0`

### 2.2 覆盖 action chunk 参数

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --execution-mode step \
  --n-action-steps 10
```

说明：

- `--action-chunk-size` 对应 checkpoint 里的 `chunk_size`，表示模型一次 forward 的动作 horizon。
- `--n-action-steps` 对应 checkpoint 里的 `n_action_steps`，表示每次推理实际返回多少步动作。
- `raw` 模式仍然只用返回 chunk 的第一个动作做逐帧对比；`step` 模式会按队列/控制环语义逐步执行。
- 对已训练好的模型，通常优先只调 `--n-action-steps`。改 `--action-chunk-size` 可能导致模型结构和权重 shape 不匹配。
- 当前 ACT checkpoint 写的是 `chunk_size=50, n_action_steps=1`；不显式传 `--n-action-steps` 时，SDK 会按真机控制兼容逻辑执行完整 50 步。显式传 `--n-action-steps 1` 时会尊重用户设置。
- 异步脚本里的 `--temporal-ensemble` 是队列层时间集成，需要 chunk overlap 才明显生效；同步 `step` 路径里的 `--temporal-ensemble` 则走 ACT 原始在线时间集成逻辑。

### 3. 验证全部 episode

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --all-episodes
```

### 4. 只跑前 50 帧做快速调试

```bash
python examples/validate_dataset_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --episode 0 \
  --max-frames 50
```

## 常用参数

- `--model`：模型目录，或 Hugging Face 模型 repo id
- `--model-type`：算法类型，支持 `act`、`pi0`、`pi05`、`smolvla`
- `--dataset`：LeRobot 数据集根目录，或 Hugging Face dataset repo id
- `--episode`：指定单个 episode
- `--all-episodes`：验证全部 episode
- `--device`：推理设备，例如 `cuda:0` 或 `cpu`
- `--instruction`：语言指令，主要用于 `pi0` / `pi05` / `smolvla`
- `--execution-mode`：`auto` / `raw` / `step`
- `--temporal-ensemble`：开启 SDK ACT 时间集成，只支持 `act` + `step`/`auto`
- `--temporal-ensemble-coeff`：ACT 时间集成系数，默认 `0.01`，必须和 `--temporal-ensemble` 一起使用
- `--action-chunk-size`：覆盖 checkpoint `chunk_size`，即模型一次 forward 的动作 horizon。通常不建议随意改已训练模型的这个值。
- `--n-action-steps`：覆盖 checkpoint `n_action_steps`，即每次推理实际返回的动作数，必须小于等于 `action_chunk_size` / `chunk_size`。
- `--enable-rtc`：为 `smolvla` / `pi0` / `pi05` 开启 RTC
- `--rtc-prefix-attention-schedule`：RTC 前缀注意力权重，支持 `ZEROS`、`ONES`、`LINEAR`、`EXP`
- `--rtc-execution-horizon` / `--rtc-inference-delay-steps`：RTC 执行窗口和静态推理延迟步数
- `--enable-async-inference`：在 `step` 模式下启动异步推理线程
- `--output-dir`：自定义输出目录
- `--dataset-gripper-scale`：夹爪值缩放模式，通常保留默认 `auto`
- `--video-backend`：LeRobotDataset 使用的视频后端，默认 `pyav`
- `--max-frames`：只处理前 N 帧，便于调试
- `--playback-mode`：仅异步验证脚本使用，`realtime` 按 FPS sleep，`fast` 只推进模拟 timestep
- `--debug-threads` / `--force-exit`：仅异步验证脚本使用，用于定位或绕过第三方库残留非 daemon 线程导致的退出阻塞

## 输出结果

如果不指定 `--output-dir`，结果默认会写到：

```bash
outputs/validate_dataset_inference/<时间戳>_<模型名>_<数据集名>_<scope>/
```

目录结构大致如下：

```text
outputs/validate_dataset_inference/20260422_170000_model_dataset_episode_000/
├── plots/
│   └── episode_000.png
├── csv/
│   └── episode_000.csv
├── summary.csv
└── summary.json
```

其中：

- `plots/episode_xxx.png`：每个动作维度的预测曲线和标注曲线
- `csv/episode_xxx.csv`：每一帧的 target / prediction / abs_error
- `summary.csv`：所有 episode 的简要统计
- `summary.json`：更完整的结构化结果

运行头部和 `summary.json` 里还会额外记录：

- 请求的执行模式和实际生效的执行模式
- 是否请求异步推理
- 运行时最终是否真的启用了异步推理

## 常见问题

### 1. 模型类型不匹配

如果你明确知道模型算法类型，建议总是传 `--model-type`，避免配置文件推断错误。

### 2. 动作维度不一致

如果脚本报错提示 `Dataset action dim ... does not match model action dim ...`，说明：

- 模型动作空间和数据集不匹配
- 或者加载了错误的 checkpoint / 错误的数据集

### 3. 缺少依赖

如果报依赖缺失，通常是环境里缺少脚本依赖，例如：

- `huggingface_hub`
- `matplotlib`
- `lerobot`
- `av`

先补齐依赖后再运行。

## 推荐命令

如果你只是想快速确认链路是否通，优先从这一条开始：

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --max-frames 20
```

如果你要专门验证控制环差异，建议直接用：

```bash
python examples/validate_dataset_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --execution-mode step \
  --max-frames 20
```
