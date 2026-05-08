# `validate_dataset_async_inference.py` 使用说明

## 作用

`examples/validate_dataset_async_inference.py` 用来通过进程内 `AsyncInferenceRuntime` 回放 LeRobot 数据集，并把控制环实际执行到的动作和数据集标注动作做逐帧对比。

它适合回答这类问题：

- 异步 runtime 能不能正常加载并启动
- 动作队列补帧策略是否合理
- chunk 重叠时的聚合策略会不会带来明显偏差
- 队列空了以后是否频繁退回 fallback action
- 在真实控制环语义下，模型和数据集动作的误差大概有多大

脚本会输出：

- 每个 episode 的动作曲线图
- 每个 episode 的逐帧 CSV
- 汇总 `summary.json` 和 `summary.csv`

和 `examples/validate_dataset_inference.py` 的区别是：

- 这个脚本只验证异步 runtime 路径，不提供 `raw` / `step` 模式切换
- 它每一帧都调用 `AsyncInferenceRuntime.step()`
- 对比 target 时，会按 runtime 返回的 `action_timestep` 对齐数据集动作，而不是简单按“当前帧”对齐

最后这一点很重要，因为异步队列里当前执行的动作，可能来自更早或更晚提交的 observation。如果不按 `action_timestep` 对齐，误差会看起来像整体错一帧。

## 前提条件

建议先进入仓库根目录并激活环境：

```bash
cd /home/synria/demo/Inference-SDK
source .venv/bin/activate
```

如果还没装完整依赖，至少需要：

```bash
pip install -e ".[all,examples]"
```

如果你依赖本地 `SparkMind` checkout，也可以额外安装：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -e ../SparkMind
```

注意：

- `SparkMind` 目前要求 Python 3.12+
- 脚本会尝试导入 `sparkmind.lerobot_compat.datasets.lerobot_dataset`
- 如果导入失败，会回退尝试 `lerobot.common.datasets.lerobot_dataset`

你还需要准备：

- 一个可加载的导出模型目录，或者 Hugging Face 模型 repo id
- 一个 LeRobot 格式的数据集根目录，或者 Hugging Face dataset repo id

## 最基本的用法

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --episode 0
```

当前支持的模型类型：

- `act`
- `smolvla`
- `pi0`
- `pi05`

如果不传 `--model-type`，脚本会尝试从模型配置自动推断。

## 运行流程

每个 episode 的验证流程大致如下：

1. 读取 episode 第一帧，构造 observation
2. 调用 `runtime.warmup()` 先同步产出一批初始动作
3. 调用 `runtime.start()` 启动后台异步推理线程
4. 对 episode 中每一帧调用一次 `runtime.step()`
5. 根据返回的 `action_timestep` 选择对应 dataset target 做误差计算
6. episode 结束后保存 plot、CSV 和汇总结果

运行时每隔一段会打印类似下面的日志：

```text
frame 0050/0180 dataset_idx=00123 target_idx=00124 action_ts=49 frame_idx=0049 source=queue queue=12 fallbacks=0 mae=0.031245
```

这些字段里最值得关注的是：

- `source=queue`：当前动作来自异步动作队列
- `source=fallback_*`：当前动作来自 fallback 逻辑，通常说明队列一度为空
- `queue=...`：当前队列里还能执行的动作数量
- `fallbacks=...`：到当前为止累计发生了多少次 fallback
- `mae=...`：当前这一帧动作向量的平均绝对误差

## 常用示例

### 1. 验证一个本地 ACT 模型

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --device cuda:0
```

### 2. 验证带时间集成的 ACT 异步控制环

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --device cuda:0 \
  --temporal-ensemble
```

如果要覆盖时间集成系数：

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --device cuda:0 \
  --temporal-ensemble \
  --temporal-ensemble-coeff 0.01
```

说明：

- `--temporal-ensemble` 只支持 `act`。
- `--temporal-ensemble-coeff` 不能单独使用，必须和 `--temporal-ensemble` 一起传。
- 这是异步 runtime 路径上的队列层时间集成：同一 timestep 被多个 chunk 预测到时，队列用 ACT 指数权重融合。
- 它需要 chunk overlap 才明显生效，通常保留默认 `n_action_steps` 或设置为大于 1；如果显式设置 `--n-action-steps 1`，异步队列里基本没有未来动作可重叠。
- 开启后，重叠 timestep 使用时间集成权重；`--aggregate-fn` 主要影响未开启时间集成时的重叠 chunk 聚合。
- 如果只想测试同步 step 路径，也可以用 `validate_dataset_inference.py --temporal-ensemble`。

### 3. 验证一个 PI0 模型并显式指定指令

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/pi0_checkpoint \
  --model-type pi0 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick up the object"
```

说明：

- `pi0`、`pi05` 和 `smolvla` 通常需要语言指令
- 如果不传 `--instruction`，脚本会优先使用数据集样本里的 `task` 字段

### 4. 验证开启 RTC 的 PI0 / PI0.5 / SmolVLA

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/pi05_checkpoint \
  --model-type pi05 \
  --dataset /path/to/lerobot_dataset \
  --episode 0 \
  --instruction "pick up the object" \
  --enable-rtc \
  --rtc-prefix-attention-schedule LINEAR \
  --rtc-execution-horizon 10 \
  --rtc-inference-delay-steps 0
```

说明：

- `--enable-rtc` 只支持 `smolvla`、`pi0`、`pi05`
- `--rtc-inference-delay-steps` 是静态控制步延迟，默认 `0`

### 5. 验证全部 episode

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --all-episodes
```

### 6. 只跑前 50 帧做快速调试

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --episode 0 \
  --max-frames 50
```

### 7. 更激进地补 action chunk

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --episode 0 \
  --chunk-size-threshold 0.8
```

说明：

- `chunk_size_threshold` 表示动作队列的填充比例阈值
- 阈值越大，runtime 越早提交新的 observation 请求下一段 chunk
- 阈值越小，runtime 越倾向于等队列更接近耗尽时再补

### 8. 覆盖 action chunk 参数

```bash
python examples/validate_dataset_async_inference.py \
  --model models/ACT_pick_and_place_v2 \
  --model-type act \
  --dataset data/lerobot/z18820636149/pick_and_place_data90 \
  --episode 0 \
  --n-action-steps 10 \
  --chunk-size-threshold 0.5
```

说明：

- `--action-chunk-size` 对应 checkpoint 里的 `chunk_size`，表示模型一次 forward 的动作 horizon。
- `--n-action-steps` 对应 checkpoint 里的 `n_action_steps`，表示每次推理实际返回并进入异步动作队列的动作数。
- `--chunk-size-threshold` 不是 action chunk 大小，而是队列 refill 阈值。判断条件是 `action_queue_size / n_action_steps <= threshold`。
- 对已训练好的模型，通常优先只调 `--n-action-steps`。改 `--action-chunk-size` 可能导致模型结构和权重 shape 不匹配。
- 当前 ACT checkpoint 写的是 `chunk_size=50, n_action_steps=1`；不显式传 `--n-action-steps` 时，SDK 会按真机控制兼容逻辑执行完整 50 步。显式传 `--n-action-steps 1` 时会尊重用户设置。

例子：如果实际 `n_action_steps=10` 且 `--chunk-size-threshold 0.5`，队列剩余 `<= 5` 个动作时会提交新 observation 触发后台推理。

### 9. 切换重叠 chunk 的聚合策略

```bash
python examples/validate_dataset_async_inference.py \
  --model /path/to/model \
  --model-type act \
  --dataset /path/to/dataset \
  --episode 0 \
  --aggregate-fn latest_only
```

当前可用的 `aggregate-fn` 有：

- `latest_only`：直接使用最新 chunk 的动作
- `weighted_average`：`0.3 * old + 0.7 * new`
- `average`：`0.5 * old + 0.5 * new`
- `conservative`：`0.7 * old + 0.3 * new`

如果你想观察控制更平滑还是更跟手，可以重点比较这里。

## 常用参数

- `--model`：模型目录，或 Hugging Face 模型 repo id
- `--model-type`：算法类型，支持 `act`、`smolvla`、`pi0`、`pi05`
- `--dataset`：LeRobot 数据集根目录，或 Hugging Face dataset repo id
- `--episode`：指定单个 episode
- `--all-episodes`：验证全部 episode
- `--device`：推理设备，例如 `cuda:0` 或 `cpu`
- `--output-dir`：自定义输出目录
- `--instruction`：显式语言指令，主要用于 `pi0` / `pi05` / `smolvla`
- `--chunk-size-threshold`：动作队列填充比例阈值，默认 `0.5`
- `--action-chunk-size`：覆盖 checkpoint `chunk_size`，即模型一次 forward 的动作 horizon。通常不建议随意改已训练模型的这个值。
- `--n-action-steps`：覆盖 checkpoint `n_action_steps`，即每次推理实际返回并进入队列的动作数，必须小于等于 `action_chunk_size` / `chunk_size`
- `--aggregate-fn`：重叠 chunk 的动作聚合策略，默认 `weighted_average`
- `--temporal-ensemble`：开启 SDK ACT 时间集成，只支持 `act`；在异步脚本里作用于 action queue 的重叠 timestep
- `--temporal-ensemble-coeff`：ACT 时间集成系数，默认 `0.01`，必须和 `--temporal-ensemble` 一起使用
- `--enable-rtc`：为 `smolvla` / `pi0` / `pi05` 开启 RTC
- `--rtc-prefix-attention-schedule`：RTC 前缀注意力权重，支持 `ZEROS`、`ONES`、`LINEAR`、`EXP`
- `--rtc-execution-horizon` / `--rtc-inference-delay-steps`：RTC 执行窗口和静态推理延迟步数
- `--dataset-gripper-scale`：夹爪值缩放模式，支持 `auto`、`normalized`、`raw`
- `--video-backend`：`LeRobotDataset` 使用的视频后端，默认 `pyav`
- `--max-frames`：只处理前 N 帧，便于调试

## 输出结果

如果不指定 `--output-dir`，结果默认会写到：

```bash
outputs/validate_dataset_inference/<时间戳>_<模型名>_<数据集名>_async_episode_000/
```

如果传了 `--all-episodes`，目录后缀会变成：

```bash
async_all_episodes
```

目录结构大致如下：

```text
outputs/validate_dataset_inference/20260430_120000_model_dataset_async_episode_000/
├── plots/
│   └── episode_000.png
├── csv/
│   └── episode_000.csv
├── summary.csv
└── summary.json
```

其中：

- `plots/episode_xxx.png`：每个动作维度的预测曲线和标注曲线
- `csv/episode_xxx.csv`：每一帧的 `target` / `prediction` / `abs_error`
- `summary.csv`：所有 episode 的简要统计
- `summary.json`：完整结构化结果

`summary.json` 里会额外记录这些异步运行信息：

- `execution_mode`，固定为 `async_runtime`
- `control_fps`
- `chunk_size_threshold`
- `aggregate_fn`
- `temporal_ensemble_enabled`
- `temporal_ensemble_coeff`
- `dataset_gripper_scale`
- 每个 episode 的 `average_step_ms`

## 常见问题

### 1. 动作维度不一致

如果报错：

```text
Dataset action dim (...) does not match model action dim (...)
```

通常说明：

- 模型动作空间和数据集不匹配
- checkpoint 选错了
- 数据集选错了

### 2. fallback 次数很多

如果日志里的 `fallbacks` 持续增长，通常表示异步队列供给不足。优先排查：

- 当前设备推理速度是否跟不上数据集 `fps`
- `--chunk-size-threshold` 是否太小
- `--n-action-steps` 是否太小，或者推理耗时太长
- 是否在 `cpu` 上跑了本来应当放在 `cuda:0` 的模型

如果只是做快速调试，也可以先降低 `--max-frames` 缩小问题范围。

### 3. `aggregate-fn` 名字无效

如果你传了未注册的名字，runtime 会报错。当前可用值只有：

- `latest_only`
- `weighted_average`
- `average`
- `conservative`

### 4. 指令模型效果异常

对于 `pi0`、`pi05` 和 `smolvla`，请先确认：

- 传入的 `--instruction` 是否和训练数据语义一致
- 数据集样本里的 `task` 字段是否正确

如果既没传 `--instruction`，数据集又没有合适的 `task`，模型输出很可能没有参考意义。

### 5. 缺少依赖

如果缺少 `huggingface_hub`、`matplotlib`、`lerobot` 或 `SparkMind`，优先执行：

```bash
pip install -e ".[all,examples]"
```

如果你需要本地 SparkMind 兼容路径，再单独处理 Python 3.12 环境。

## 建议的使用顺序

如果你是第一次排查某个模型的异步控制环行为，建议按下面顺序来：

1. 先只跑一个 episode
2. 再加 `--max-frames 20` 或 `--max-frames 50`
3. 看日志里的 `source`、`queue`、`fallbacks`
4. 再调 `--n-action-steps` 和 `--chunk-size-threshold`
5. 最后再比较不同 `--aggregate-fn` 的误差和曲线差异

这样更容易区分问题是出在模型本身，还是出在异步调度策略上。
