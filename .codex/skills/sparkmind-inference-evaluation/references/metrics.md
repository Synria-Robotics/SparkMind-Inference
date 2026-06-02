# Metrics And Acceptance Standards

## Official Assets

Use these assets for correctness claims:

| Model family | Checkpoint | Dataset | Primary purpose |
| --- | --- | --- | --- |
| SmolVLA | `lerobot/smolvla_libero` | `lerobot/libero` | LIBERO offline and closed-loop validation |
| PI0 | `lerobot/pi0_libero_base` | `lerobot/libero` | PI0 SDK-vs-LeRobot action alignment |
| PI0.5 base | `lerobot/pi05_libero_base` | `lerobot/libero` | PI0.5 SDK-vs-LeRobot action alignment |
| PI0.5 final | `lerobot/pi05-libero` | `lerobot/libero` | PI0.5 LIBERO closed-loop validation |
| ACT | `lerobot/act_aloha_sim_transfer_cube_human` | `lerobot/aloha_sim_transfer_cube_human` | ACT raw and temporal ensemble alignment |

Do not use private or ad-hoc checkpoints as evidence for LeRobot official correctness.

## Core Offline Metrics

Read from `summary.json` and `comparison.csv`:

- `first_action.mae`: main smoke metric for chunk policy alignment.
- `first_action.rmse`: useful when a few dimensions dominate.
- `first_action.max_abs`: catches gripper/unit explosions and single-dim key mismatches.
- `per_dim_mae`: localizes camera/state/gripper errors by action dimension.
- `full_chunk.mae`: use with `--compare-full-chunk` when debugging horizon/chunk behavior.

Healthy comparator results should be deterministic for a fixed seed. If a regression appears, rerun once before debugging.

## Acceptance Thresholds

Use these as current project acceptance signals:

| Path | Healthy signal | Failure signal |
| --- | --- | --- |
| PI0 no-RTC, `lerobot/pi0_libero_base` | first-action MAE about `4e-4`; max_abs about `1e-3` | MAE `>1e-2` |
| PI0.5 no-RTC, `lerobot/pi05_libero_base` | first-action MAE about `2.5e-4`; max_abs below `1e-3` | MAE `>1e-2` |
| PI0.5 no-RTC, `lerobot/pi05-libero` | first-action MAE around `1e-3` | MAE `>2e-2` |
| PI RTC chunk | MAE should stay near no-RTC scale to low `1e-3`, depending on horizon/delay | MAE jumps by orders of magnitude |
| ACT raw chunk | MAE around `1e-5` or lower | MAE `>1e-3` |
| ACT temporal ensemble | MAE around `1e-5` or lower with `n_action_steps=1` and preserved policy state | MAE `>1e-3` |
| SmolVLA offline | first-action MAE should be small; full closed-loop confidence comes from simulator comparator | persistent large per-dim mismatch |
| SmolVLA `libero_spatial` task 0 simulator | SDK FIFO rollout succeeds; official trajectory + SDK action diff near `0.02` MAE | SDK FIFO fails while official succeeds and action diff is large |

Observed recent smoke results after PI streaming/meta loading:

| Path | first-action MAE | max_abs |
| --- | ---: | ---: |
| `lerobot/pi0_libero_base`, no-RTC, 1 frame | `0.00043947` | `0.00127262` |
| `lerobot/pi05_libero_base`, no-RTC, 1 frame | `0.00024842` | `0.00064921` |

## Performance Metrics

Report these for load/inference optimization work:

- `load_time_s`
- `rss_peak_after_load_mib`
- `rss_after_load_mib`
- `load_peak_alloc_mib`
- `load_peak_reserved_mib`
- `predict_first_ms`
- `predict_steady_mean_ms`
- `predict_peak_alloc_mib`
- `predict_peak_reserved_mib`

Recent PI loading targets after meta + streaming loading:

| Model | load time | CPU RSS peak | CPU RSS after load | GPU load peak | steady predict |
| --- | ---: | ---: | ---: | ---: | ---: |
| PI0 | `~11.66s` | `~7.75GB` | `~1.32GB` | `~16.0GB` | `~374ms` |
| PI0.5 | `~11.65s` | `~15.1GB` | `~1.31GB` | `~15.45GB` | `~404ms` |

For PI0/PI0.5 loading, a return to `~40GB` CPU RSS peak usually means the code is materializing the model on CPU or loading the whole safetensors state dict at once.

## Failure Pattern Cheat Sheet

- PI0/PI0.5 MAE jumps to `~0.7`: incorrect state/action normalization ownership.
- PI0.5 MAE jumps to `~0.04`: meta initialization left non-persistent runtime buffers uninitialized, especially vision `position_ids` or Gemma rotary `inv_freq`.
- ACT last action dimension explodes: gripper scaling was applied to a non-7D ACT/Aloha action.
- SmolVLA simulator timestamp mode diverges while FIFO is healthy: queue timing mismatch, not necessarily model forward mismatch.
- Missing camera features in PI checkpoints: pass `-1` image tensor and false image mask, matching LeRobot behavior.
- RTC enabled with `predict_action()` / `engine.step()`: invalid for PI/SmolVLA; use `predict_action_chunk()` and an RTC-aware upper action queue.

