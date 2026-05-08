#!/usr/bin/env python3
"""Validate dataset episodes through the process-local async inference runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from validate_dataset_inference import (
    EngineMetadata,
    ErrorAccumulator,
    EpisodeResult,
    _adapt_prediction_for_dataset,
    _build_episode_to_indices,
    _build_observation,
    _dataset_repo_id_for_source,
    _default_dataset_source,
    _default_device,
    _default_model_source,
    _default_output_dir,
    _detect_dataset_gripper_mode,
    _load_action_metadata,
    _load_dataset_class,
    _plot_episode_curves,
    _resolve_dataset_source,
    _resolve_instruction,
    _resolve_model_source,
    _resolve_model_type,
    _select_episode_ids,
    _write_episode_csv,
    _write_summary_csv,
)
from inference_sdk import AsyncInferenceConfig, AsyncInferenceRuntime, SUPPORTED_MODEL_TYPES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SDK inference against a LeRobot dataset using only "
            "AsyncInferenceRuntime."
        )
    )
    parser.add_argument(
        "--model",
        default=_default_model_source(),
        help="Local exported model directory or Hugging Face model repo id",
    )
    parser.add_argument(
        "--model-type",
        type=str.lower,
        choices=SUPPORTED_MODEL_TYPES,
        default=None,
        help="Explicit model type override. Default: infer from model config.",
    )
    parser.add_argument(
        "--dataset",
        default=_default_dataset_source(),
        help="Local LeRobot dataset root or Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to validate. Ignored when --all-episodes is used.",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Validate every episode and generate one plot per episode.",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Inference device. Default: cuda:0 when available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots and reports. Default: outputs/validate_dataset_inference/<timestamp>_...",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Optional explicit instruction. PI0/SmolVLA otherwise use the dataset task string.",
    )
    parser.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=0.5,
        help="Action queue fill-ratio threshold for submitting new observations.",
    )
    parser.add_argument(
        "--action-chunk-size",
        type=int,
        default=None,
        help="Override checkpoint chunk_size, the model forward action horizon.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override checkpoint n_action_steps, the number of actions returned/enqueued per inference.",
    )
    parser.add_argument(
        "--aggregate-fn",
        default="weighted_average",
        help="Aggregation strategy for overlapping async action chunks.",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action="store_true",
        help="Enable SDK ACT temporal ensembling.",
    )
    parser.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="ACT temporal ensembling coefficient. Default: 0.01 when --temporal-ensemble is set.",
    )
    parser.add_argument(
        "--enable-rtc",
        action="store_true",
        help="Enable RTC for SmolVLA/PI0/PI0.5 policies.",
    )
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=["ZEROS", "ONES", "LINEAR", "EXP", "zeros", "ones", "linear", "exp"],
        default="LINEAR",
        help="RTC prefix attention schedule. Default: LINEAR.",
    )
    parser.add_argument(
        "--rtc-max-guidance-weight",
        type=float,
        default=10.0,
        help="RTC max guidance weight. Default: 10.0.",
    )
    parser.add_argument(
        "--rtc-execution-horizon",
        type=int,
        default=10,
        help="RTC execution horizon. Default: 10.",
    )
    parser.add_argument(
        "--rtc-inference-delay-steps",
        type=int,
        default=0,
        help="Static RTC inference delay in control steps. Default: 0.",
    )
    parser.add_argument(
        "--rtc-debug",
        action="store_true",
        help="Enable RTC debug tracking in SparkMind's RTCProcessor.",
    )
    parser.add_argument(
        "--rtc-debug-maxlen",
        type=int,
        default=100,
        help="Maximum RTC debug records to keep. Default: 100.",
    )
    parser.add_argument(
        "--playback-mode",
        choices=["offline", "fast", "realtime"],
        default="offline",
        help=(
            "Validation playback mode. `offline` runs one synchronous chunk prediction per dataset frame "
            "and compares chunk[0] against that frame; `fast`/`realtime` exercise the live async queue."
        ),
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for the warmup action queue before entering the loop.",
    )
    parser.add_argument(
        "--disable-gripper-clamping",
        action="store_true",
        help="Disable runtime gripper velocity clamping for curve comparison.",
    )
    parser.add_argument(
        "--debug-threads",
        action="store_true",
        help="Print remaining non-daemon Python threads before exit.",
    )
    parser.add_argument(
        "--force-exit",
        action="store_true",
        help="Force os._exit(0) after reports are flushed; useful if third-party libraries leave non-daemon threads.",
    )
    parser.add_argument(
        "--dataset-gripper-scale",
        choices=["auto", "normalized", "raw"],
        default="auto",
        help="How the dataset stores gripper values. Usually auto is correct.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="Video backend used by LeRobotDataset. Default: pyav.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit for the number of frames processed per episode.",
    )
    return parser.parse_args()


def _metadata_from_policy_metadata(metadata: Any) -> EngineMetadata:
    return EngineMetadata(
        required_cameras=list(metadata.required_cameras),
        action_dim=int(metadata.action_dim),
        chunk_size=int(metadata.chunk_size),
        n_action_steps=int(metadata.n_action_steps),
    )


def _load_runtime(
    model_type: str,
    model_dir: Path,
    args: argparse.Namespace,
    control_fps: float,
) -> tuple[AsyncInferenceRuntime, EngineMetadata]:
    runtime = AsyncInferenceRuntime()
    metadata = runtime.load_policy(
        algorithm_type=model_type,
        checkpoint_dir=str(model_dir),
        device=args.device,
        instruction=args.instruction,
        config=AsyncInferenceConfig(
            control_fps=control_fps,
            chunk_size_threshold=args.chunk_size_threshold,
            action_chunk_size=args.action_chunk_size,
            n_action_steps=args.n_action_steps,
            aggregate_fn_name=args.aggregate_fn,
            enable_temporal_ensemble=args.temporal_ensemble,
            temporal_ensemble_coeff=(
                0.01 if args.temporal_ensemble and args.temporal_ensemble_coeff is None
                else (args.temporal_ensemble_coeff or 0.01)
            ),
            enable_gripper_clamping=not args.disable_gripper_clamping,
            enable_rtc=args.enable_rtc,
            rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
            rtc_max_guidance_weight=args.rtc_max_guidance_weight,
            rtc_execution_horizon=args.rtc_execution_horizon,
            rtc_inference_delay_steps=args.rtc_inference_delay_steps,
            rtc_debug=args.rtc_debug,
            rtc_debug_maxlen=args.rtc_debug_maxlen,
        ),
    )
    return runtime, _metadata_from_policy_metadata(metadata)


def _sleep_until_dataset_tick(start_time: float, timestep: int, fps: float) -> None:
    target_time = start_time + (timestep + 1) / fps
    time.sleep(max(0.0, target_time - time.monotonic()))


def _cleanup_process_resources(*, debug_threads: bool) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()

    if debug_threads:
        current = threading.current_thread()
        remaining = [
            thread
            for thread in threading.enumerate()
            if thread is not current and thread.is_alive() and not thread.daemon
        ]
        if remaining:
            print("Remaining non-daemon threads:")
            for thread in remaining:
                print(f"  name={thread.name!r} ident={thread.ident}")
        else:
            print("Remaining non-daemon threads: none")


def _print_run_header(
    *,
    model_type: str,
    model_dir: Path,
    model_label: str,
    dataset_root: Path,
    dataset_label: str,
    dataset: Any,
    metadata: EngineMetadata,
    gripper_mode: str,
    episode_ids: list[int],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    print("=" * 80)
    print("SDK Async Runtime Dataset Validation")
    print("=" * 80)
    print(f"Model type: {model_type}")
    print(f"Model source: {model_label}")
    print(f"Model dir: {model_dir}")
    print(f"Dataset source: {dataset_label}")
    print(f"Dataset root: {dataset_root}")
    print(f"Frames: {len(dataset)}  Episodes: {dataset.num_episodes}  FPS: {dataset.fps}")
    print(f"Required cameras: {metadata.required_cameras}")
    print(f"Action dim: {metadata.action_dim}  Chunk size: {metadata.chunk_size}")
    print(f"Dataset gripper scale: {gripper_mode}")
    print(f"Device: {args.device}")
    print("Execution mode: async runtime" if args.playback_mode != "offline" else "Execution mode: offline chunk validation")
    print(f"Chunk threshold: {args.chunk_size_threshold}")
    print(f"Aggregate fn: {args.aggregate_fn}")
    print(f"Temporal ensemble: {'enabled' if args.temporal_ensemble else 'disabled'}")
    print(f"RTC: {'enabled' if args.enable_rtc else 'disabled'}")
    print(f"Playback mode: {args.playback_mode}")
    print(f"Gripper clamping: {'disabled' if args.disable_gripper_clamping else 'enabled'}")
    print(f"Episodes to validate: {episode_ids}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)


def _run_async_episode_validation(
    *,
    runtime: AsyncInferenceRuntime,
    dataset: Any,
    dataset_indices: np.ndarray,
    all_actions: np.ndarray,
    metadata: EngineMetadata,
    model_type: str,
    gripper_mode: str,
    explicit_instruction: str | None,
    args: argparse.Namespace,
) -> EpisodeResult:
    action_dim = int(metadata.action_dim)
    accumulator = ErrorAccumulator(action_dim=action_dim)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frame_indices: list[int] = []
    task = ""
    total_call_ms = 0.0
    started = False

    runtime.reset(clear_metrics=False)
    first_item = dataset[int(dataset_indices[0])]
    episode_instruction = _resolve_instruction(model_type, explicit_instruction, first_item)
    simulation_start = time.monotonic()
    dt = 1.0 / float(dataset.fps)

    try:
        first_observation = _build_observation(
            item=first_item,
            required_cameras=list(metadata.required_cameras),
            gripper_mode=gripper_mode,
            instruction=episode_instruction,
        )
        runtime.warmup(
            images=first_observation.images,
            state=first_observation.state,
            instruction=episode_instruction,
            timestamp=simulation_start,
            timestep=0,
        )
        runtime.start()
        started = True
        if not runtime.wait_until_ready(min_queue_size=1, timeout=args.startup_timeout):
            status = runtime.get_status()
            raise RuntimeError(f"Async runtime did not become ready: state={status.state} error={status.last_error}")

        for position, dataset_index in enumerate(dataset_indices.tolist(), start=1):
            control_timestep = position - 1
            control_timestamp = simulation_start + control_timestep * dt
            item = dataset[int(dataset_index)]
            instruction = _resolve_instruction(model_type, explicit_instruction, item) or episode_instruction
            observation = _build_observation(
                item=item,
                required_cameras=list(metadata.required_cameras),
                gripper_mode=gripper_mode,
                instruction=instruction,
            )

            start_time = time.perf_counter()
            result = runtime.step(
                images=observation.images,
                state=observation.state,
                instruction=instruction,
                timestamp=control_timestamp,
                timestep=control_timestep,
            )
            total_call_ms += (time.perf_counter() - start_time) * 1000.0

            # Async runtime returns the action that should execute at the current
            # control timestep, which may differ from the just-submitted frame.
            # Compare against the dataset action with the same action timestep.
            target_position = result.action_timestep
            if target_position is None or target_position < 0 or target_position >= len(dataset_indices):
                target_position = position - 1
            target_dataset_index = int(dataset_indices[int(target_position)])

            prediction = _adapt_prediction_for_dataset(result.action, gripper_mode)[0]
            target = all_actions[target_dataset_index]
            accumulator.update(prediction, target)

            predictions.append(prediction)
            targets.append(target)

            frame_value = item["frame_index"]
            if hasattr(frame_value, "item"):
                frame_value = frame_value.item()
            frame_indices.append(int(frame_value))
            task = str(item.get("task", task))

            if position == 1 or position == len(dataset_indices) or position % 50 == 0:
                status = runtime.get_status()
                sample_mae = float(np.mean(np.abs(prediction - target)))
                print(
                    f"  frame {position:04d}/{len(dataset_indices):04d} "
                    f"dataset_idx={dataset_index:05d} target_idx={target_dataset_index:05d} "
                    f"action_ts={result.action_timestep} frame_idx={frame_indices[-1]:04d} "
                    f"source={result.source} queue={result.queue_size} "
                    f"fallbacks={status.fallback_count} dropped_obs={status.dropped_observation_count} "
                    f"infer_ms={status.last_inference_ms} mae={sample_mae:.6f}"
                )
            if args.playback_mode == "realtime":
                _sleep_until_dataset_tick(simulation_start, control_timestep, float(dataset.fps))
    finally:
        if started:
            runtime.stop()

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    metrics = accumulator.as_dict()
    episode_value = dataset.hf_dataset[int(dataset_indices[0])]["episode_index"]
    if hasattr(episode_value, "item"):
        episode_value = episode_value.item()

    return EpisodeResult(
        episode_index=int(episode_value),
        task=task,
        dataset_indices=dataset_indices.copy(),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        predictions=predictions_array,
        targets=targets_array,
        metrics=metrics,
        average_call_ms=total_call_ms / max(1, len(dataset_indices)),
    )


def _run_offline_episode_validation(
    *,
    runtime: AsyncInferenceRuntime,
    dataset: Any,
    dataset_indices: np.ndarray,
    all_actions: np.ndarray,
    metadata: EngineMetadata,
    model_type: str,
    gripper_mode: str,
    explicit_instruction: str | None,
) -> EpisodeResult:
    action_dim = int(metadata.action_dim)
    accumulator = ErrorAccumulator(action_dim=action_dim)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    frame_indices: list[int] = []
    task = ""
    total_call_ms = 0.0

    runtime.reset(clear_metrics=False)
    first_item = dataset[int(dataset_indices[0])]
    episode_instruction = _resolve_instruction(model_type, explicit_instruction, first_item)

    for position, dataset_index in enumerate(dataset_indices.tolist(), start=1):
        item = dataset[int(dataset_index)]
        instruction = _resolve_instruction(model_type, explicit_instruction, item) or episode_instruction
        observation = _build_observation(
            item=item,
            required_cameras=list(metadata.required_cameras),
            gripper_mode=gripper_mode,
            instruction=instruction,
        )

        start_time = time.perf_counter()
        chunk = runtime.predict_action_chunk(
            images=observation.images,
            state=observation.state,
            instruction=instruction,
        )
        total_call_ms += (time.perf_counter() - start_time) * 1000.0

        prediction = _adapt_prediction_for_dataset(chunk[0], gripper_mode)[0]
        target = all_actions[int(dataset_index)]
        accumulator.update(prediction, target)

        predictions.append(prediction)
        targets.append(target)

        frame_value = item["frame_index"]
        if hasattr(frame_value, "item"):
            frame_value = frame_value.item()
        frame_indices.append(int(frame_value))
        task = str(item.get("task", task))

        if position == 1 or position == len(dataset_indices) or position % 50 == 0:
            sample_mae = float(np.mean(np.abs(prediction - target)))
            print(
                f"  frame {position:04d}/{len(dataset_indices):04d} "
                f"dataset_idx={dataset_index:05d} frame_idx={frame_indices[-1]:04d} "
                f"infer_ms={total_call_ms / position:.3f} mae={sample_mae:.6f}"
            )

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    metrics = accumulator.as_dict()
    episode_value = dataset.hf_dataset[int(dataset_indices[0])]["episode_index"]
    if hasattr(episode_value, "item"):
        episode_value = episode_value.item()

    return EpisodeResult(
        episode_index=int(episode_value),
        task=task,
        dataset_indices=dataset_indices.copy(),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        predictions=predictions_array,
        targets=targets_array,
        metrics=metrics,
        average_call_ms=total_call_ms / max(1, len(dataset_indices)),
    )


def main() -> int:
    args = _parse_args()
    scope = "all_episodes" if args.all_episodes else f"episode_{args.episode:03d}"
    output_dir = Path(
        args.output_dir or _default_output_dir(args.model, args.dataset, f"async_{scope}")
    ).expanduser().resolve()
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    model_dir, model_label = _resolve_model_source(args.model)
    dataset_root, dataset_label = _resolve_dataset_source(args.dataset)
    model_type = _resolve_model_type(args.model_type, model_dir)
    if args.temporal_ensemble and model_type != "act":
        raise ValueError("`--temporal-ensemble` is only supported for ACT")
    if args.temporal_ensemble_coeff is not None and not args.temporal_ensemble:
        raise ValueError("`--temporal-ensemble-coeff` requires `--temporal-ensemble`")
    if args.enable_rtc and model_type not in {"smolvla", "pi0", "pi05"}:
        raise ValueError("`--enable-rtc` is only supported for SmolVLA, PI0 and PI0.5")
    if args.rtc_max_guidance_weight <= 0.0:
        raise ValueError("`--rtc-max-guidance-weight` must be > 0")
    if args.rtc_execution_horizon < 1:
        raise ValueError("`--rtc-execution-horizon` must be >= 1")
    if args.rtc_inference_delay_steps < 0:
        raise ValueError("`--rtc-inference-delay-steps` must be >= 0")
    if args.rtc_debug_maxlen < 1:
        raise ValueError("`--rtc-debug-maxlen` must be >= 1")

    dataset_repo_id = _dataset_repo_id_for_source(args.dataset, dataset_root)

    LeRobotDataset = _load_dataset_class()
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=str(dataset_root),
        video_backend=args.video_backend,
    )
    all_actions, all_episode_indices = _load_action_metadata(dataset)
    gripper_mode = _detect_dataset_gripper_mode(all_actions, args.dataset_gripper_scale)

    episode_to_indices = _build_episode_to_indices(all_episode_indices)
    available_episode_ids = sorted(episode_to_indices.keys())
    episode_ids = _select_episode_ids(args, available_episode_ids)

    runtime, metadata = _load_runtime(model_type, model_dir, args, float(dataset.fps))
    try:
        if all_actions.shape[1] != metadata.action_dim:
            raise ValueError(
                f"Dataset action dim ({all_actions.shape[1]}) does not match model action dim ({metadata.action_dim})"
            )

        _print_run_header(
            model_type=model_type,
            model_dir=model_dir,
            model_label=model_label,
            dataset_root=dataset_root,
            dataset_label=dataset_label,
            dataset=dataset,
            metadata=metadata,
            gripper_mode=gripper_mode,
            episode_ids=episode_ids,
            output_dir=output_dir,
            args=args,
        )

        episode_results: list[EpisodeResult] = []
        overall_accumulator = ErrorAccumulator(action_dim=int(metadata.action_dim))

        for episode_index in episode_ids:
            episode_indices = episode_to_indices[episode_index]
            if args.max_frames is not None:
                episode_indices = episode_indices[: max(0, args.max_frames)]
            if len(episode_indices) == 0:
                print(f"Episode {episode_index} skipped: no frames selected")
                print("-" * 80)
                continue

            print(f"Episode {episode_index} start: {len(episode_indices)} frames")
            if args.playback_mode == "offline":
                result = _run_offline_episode_validation(
                    runtime=runtime,
                    dataset=dataset,
                    dataset_indices=episode_indices,
                    all_actions=all_actions,
                    metadata=metadata,
                    model_type=model_type,
                    gripper_mode=gripper_mode,
                    explicit_instruction=args.instruction,
                )
            else:
                result = _run_async_episode_validation(
                    runtime=runtime,
                    dataset=dataset,
                    dataset_indices=episode_indices,
                    all_actions=all_actions,
                    metadata=metadata,
                    model_type=model_type,
                    gripper_mode=gripper_mode,
                    explicit_instruction=args.instruction,
                    args=args,
                )

            for prediction, target in zip(result.predictions, result.targets, strict=True):
                overall_accumulator.update(prediction, target)

            plot_path = plots_dir / f"episode_{episode_index:03d}.png"
            csv_path = csv_dir / f"episode_{episode_index:03d}.csv"
            _plot_episode_curves(
                episode_result=result,
                model_label=model_label,
                dataset_label=dataset_label,
                output_path=plot_path,
            )
            _write_episode_csv(result, csv_path)

            result.plot_path = str(plot_path)
            result.csv_path = str(csv_path)
            episode_results.append(result)

            print(
                f"Episode {episode_index} done: frames={len(result.frame_indices)} "
                f"mae={result.metrics.get('mae', float('nan')):.6f} "
                f"rmse={result.metrics.get('rmse', float('nan')):.6f} "
                f"max_abs={result.metrics.get('max_abs', float('nan')):.6f} "
                f"avg_step_ms={result.average_call_ms:.2f} "
                f"plot={plot_path.name}"
            )
            if args.playback_mode == "offline":
                print("Runtime stats: offline direct chunks, no async queue/drop/fallback metrics")
            else:
                status = runtime.get_status()
                print(
                    "Runtime stats: "
                    f"inferences={status.inference_count} processed_obs={status.processed_observation_count} "
                    f"dropped_obs={status.dropped_observation_count} skipped_obs={status.skipped_observation_count} "
                    f"fallbacks={status.fallback_count} last_infer_ms={status.last_inference_ms}"
                )
            print("-" * 80)
    finally:
        runtime.close()

    overall_metrics = overall_accumulator.as_dict()
    summary = {
        "model_type": model_type,
        "model_source": model_label,
        "model_dir": str(model_dir),
        "dataset_source": dataset_label,
        "dataset_root": str(dataset_root),
        "device": args.device,
        "control_fps": float(dataset.fps),
        "chunk_size_threshold": args.chunk_size_threshold,
        "aggregate_fn": args.aggregate_fn,
        "temporal_ensemble_enabled": args.temporal_ensemble,
        "temporal_ensemble_coeff": (
            0.01 if args.temporal_ensemble and args.temporal_ensemble_coeff is None
            else args.temporal_ensemble_coeff
        ),
        "rtc_enabled": args.enable_rtc,
        "rtc_prefix_attention_schedule": args.rtc_prefix_attention_schedule,
        "rtc_max_guidance_weight": args.rtc_max_guidance_weight,
        "rtc_execution_horizon": args.rtc_execution_horizon,
        "rtc_inference_delay_steps": args.rtc_inference_delay_steps,
        "playback_mode": args.playback_mode,
        "gripper_clamping_enabled": not args.disable_gripper_clamping,
        "dataset_gripper_scale": gripper_mode,
        "execution_mode": "offline_chunk_validation" if args.playback_mode == "offline" else "async_runtime",
        "episodes": [
            {
                "episode_index": result.episode_index,
                "task": result.task,
                "num_frames": len(result.frame_indices),
                "metrics": result.metrics,
                "average_step_ms": result.average_call_ms,
                "plot_path": result.plot_path,
                "csv_path": result.csv_path,
            }
            for result in episode_results
        ],
        "overall": overall_metrics,
    }

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_summary_csv(episode_results, output_dir / "summary.csv")

    print()
    print("=" * 80)
    print("Async Runtime Summary")
    print("=" * 80)
    print(f"Episodes validated: {[result.episode_index for result in episode_results]}")
    print(
        f"overall mae={overall_metrics.get('mae', float('nan')):.6f} "
        f"rmse={overall_metrics.get('rmse', float('nan')):.6f} "
        f"max_abs={overall_metrics.get('max_abs', float('nan')):.6f}"
    )
    print(f"Plots dir: {plots_dir}")
    print(f"Episode CSV dir: {csv_dir}")
    print(f"Summary JSON: {summary_json_path}")

    _cleanup_process_resources(debug_threads=args.debug_threads)
    if args.force_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
