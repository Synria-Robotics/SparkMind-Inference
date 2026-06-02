---
name: sparkmind-inference-evaluation
description: Validate SparkMind Inference SDK correctness, simulator behavior, release readiness, and load/inference performance for ACT, SmolVLA, PI0, and PI0.5 against official LeRobot checkpoints, datasets, metrics, and known acceptance thresholds.
---

# SparkMind Inference Evaluation

Use this skill when checking whether `sparkmind-inference` matches LeRobot/SparkMind behavior, when debugging action mismatch, when validating a release candidate, or when reporting load/inference performance.

## Ground Rules

- Use official LeRobot checkpoints and datasets for correctness claims.
- Prefer offline SDK-vs-official action comparison before simulator debugging.
- Treat simulator success rate as task-level evidence; inspect action MAE/max_abs first when a rollout fails.
- Use the Python 3.13 LeRobot v0.5.1 environment unless the user explicitly asks for another environment:

```bash
/home/ytyang/.venvs/sparkmind313/bin/python
```

- For CUDA validation on this machine, prefer:

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

- When Hugging Face download is needed on the CUHK campus network, use:

```bash
HTTP_PROXY=http://proxy.cse.cuhk.edu.hk:8000 \
HTTPS_PROXY=http://proxy.cse.cuhk.edu.hk:8000 \
ALL_PROXY=http://proxy.cse.cuhk.edu.hk:8000
```

## Workflow

1. Run static checks: `compileall` and unit tests.
2. Run offline SDK-vs-official comparator for the target model.
3. For ACT temporal ensemble or PI RTC, run the specific mode comparator.
4. For LIBERO task behavior, run the LeRobot env wrapper simulator comparator and save video when useful.
5. For performance changes, run load/predict benchmarks and report CPU RSS, GPU peak allocated/reserved, load time, first latency, and steady latency.
6. Compare results against [metrics.md](references/metrics.md).

## Canonical Commands

Static checks:

```bash
/home/ytyang/.venvs/sparkmind313/bin/python -m compileall sparkmind_inference examples tests
/home/ytyang/.venvs/sparkmind313/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Offline comparator:

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
/home/ytyang/.venvs/sparkmind313/bin/python examples/compare_lerobot_official_inference.py \
  --model MODEL \
  --model-type MODEL_TYPE \
  --dataset DATASET \
  --episode 0 \
  --max-frames 3 \
  --device cuda:0 \
  --output-dir outputs/official_compare/NAME
```

ACT temporal ensemble comparator:

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
/home/ytyang/.venvs/sparkmind313/bin/python examples/compare_lerobot_official_inference.py \
  --model lerobot/act_aloha_sim_transfer_cube_human \
  --model-type act \
  --dataset lerobot/aloha_sim_transfer_cube_human \
  --episode 0 \
  --max-frames 5 \
  --device cuda:0 \
  --dataset-gripper-scale raw \
  --temporal-ensemble \
  --temporal-ensemble-coeff 0.01 \
  --preserve-policy-state \
  --output-dir outputs/official_compare/act_temporal_ensemble_ep0_5frames
```

PI RTC chunk comparator:

```bash
CUDA_VISIBLE_DEVICES=4 PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
/home/ytyang/.venvs/sparkmind313/bin/python examples/compare_lerobot_official_inference.py \
  --model lerobot/pi0_libero_base \
  --model-type pi0 \
  --dataset lerobot/libero \
  --episode 0 \
  --max-frames 5 \
  --device cuda:0 \
  --enable-rtc \
  --preserve-policy-state \
  --rtc-execution-horizon 10 \
  --rtc-inference-delay-steps 0 \
  --output-dir outputs/official_compare/pi0_rtc_ep0_5frames
```

LIBERO simulator comparator:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=4 \
/home/ytyang/.venvs/sparkmind313/bin/python examples/compare_libero_sim_actions.py \
  --model lerobot/smolvla_libero \
  --benchmark libero_spatial \
  --task-id 0 \
  --max-steps 280 \
  --device cuda:0 \
  --execute sdk \
  --sdk-selection fifo \
  --save-video \
  --output-dir outputs/video_compare/sdk_fifo_task0
```

## Reporting

Always report:

- checkpoint, dataset, episode, device, environment version if relevant.
- `first_action.mae`, `first_action.max_abs`, and full-chunk metrics if enabled.
- simulator `success`, episode length/return, and video path when generated.
- performance: load time, CPU RSS peak, GPU peak allocated/reserved, first predict latency, steady predict latency.
- whether results match [metrics.md](references/metrics.md) acceptance thresholds.

