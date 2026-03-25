"""Single-task steering vector pipeline for LIBERO-Spatial.

Optimized for repeated single-instruction analysis:
  Phase 0 – Prepare:  Cache matched task samples locally once.
  Phase 1 – Collect:  Forward pass over cached samples, save residuals / preds / losses.
  Phase 2 – Vectors:  Compute steering vectors from stored residuals.
  Phase 3 – Spearman: Cosine-similarity vs loss correlation.
  Phase 4 – Steer:    Steered inference over the cached samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

from experiments.analysis.libero_sira_correlation import (
    ACTION_DIM,
    DEVICE,
    DTYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    build_dataloader,
    build_prompt_only_inputs,
    compute_num_patches,
    extract_predictive_hidden_from_layer,
    infer_num_prompt_tokens,
    resolve_llm_layers,
    unnormalize_bounds_tensor,
)
from experiments.analysis.libero_steering_eval import CaptureHooks, MultiLayerSteeringHooks
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla


SUBSET_CACHE_VERSION = 1
DEFAULT_VECTOR_TASK_INSTRUCTION = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"
DEFAULT_EVAL_TASK_INSTRUCTION = "pick up the black bowl next to the cookie box and place it on the plate"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-task steering: prepare -> collect -> vectors -> spearman -> steer.")
    p.add_argument("--data-root-dir", required=True, help="Path to RLDS datasets directory")
    p.add_argument("--dataset-name", default="libero_spatial_no_noops")
    p.add_argument("--pretrained-checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
    p.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    p.add_argument(
        "--task-instruction",
        default=None,
        help="Shared fallback task instruction. If vector/eval task instructions are not provided, both use this value.",
    )
    p.add_argument(
        "--vector-task-instruction",
        default=DEFAULT_VECTOR_TASK_INSTRUCTION,
        help="Task instruction used to build the cached subset and steering vector.",
    )
    p.add_argument(
        "--eval-task-instruction",
        default=DEFAULT_EVAL_TASK_INSTRUCTION,
        help="Task instruction used for steered evaluation. If omitted, the script picks a different task by default.",
    )
    p.add_argument("--max-samples", type=int, default=400)
    p.add_argument(
        "--max-scan",
        type=int,
        default=50000,
        help="Maximum raw dataset samples to inspect when auto-discovering the target instruction.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--coeff", type=float, default=0.1, help="Steering coefficient")
    p.add_argument("--preserve-norm", action="store_true", default=False)
    p.add_argument("--phase", choices=["all", "prepare", "collect", "vectors", "spearman", "steer"], default="all")
    p.add_argument("--output-dir", default="outputs/libero_single_task_steering")
    p.add_argument("--center-crop", action="store_true", default=True)
    p.add_argument("--rebuild-subset", action="store_true", default=False, help="Ignore any cached task subset and rebuild it.")
    return p.parse_args()


def make_cfg(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=args.center_crop,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        lora_rank=32,
        unnorm_key=args.unnorm_key,
        load_in_8bit=False,
        load_in_4bit=False,
    )


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def canonicalize_instruction(instruction: Optional[str]) -> Optional[str]:
    if instruction is None:
        return None
    normalized = instruction.strip().lower()
    return normalized or None


def resolve_task_instructions(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    shared_task = canonicalize_instruction(args.task_instruction)
    vector_task = canonicalize_instruction(args.vector_task_instruction) or shared_task
    eval_task = canonicalize_instruction(args.eval_task_instruction) or shared_task or vector_task
    return vector_task, eval_task


def decode_instruction_from_input_ids(input_ids: torch.Tensor, tokenizer) -> str:
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    marker = "What action should the robot take to "
    idx = text.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = text.find("?", start)
    if end == -1:
        return text[start:].strip().lower()
    return text[start:end].strip().lower()


def get_batch_instruction(batch: Dict[str, object], tokenizer) -> str:
    if "task_instructions" in batch and batch["task_instructions"]:
        return str(batch["task_instructions"][0]).strip().lower()
    return decode_instruction_from_input_ids(batch["input_ids"], tokenizer)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def task_slug(instruction: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", instruction).strip("-")[:80] or "task"
    digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def get_subset_dir(output_dir: Path, instruction: str) -> Path:
    return output_dir / "subset" / task_slug(instruction)


def get_subset_metadata_path(subset_dir: Path) -> Path:
    return subset_dir / "metadata.json"


def get_subset_samples_dir(subset_dir: Path) -> Path:
    return subset_dir / "samples"


def load_subset_metadata(subset_dir: Path) -> dict:
    with open(get_subset_metadata_path(subset_dir)) as f:
        return json.load(f)


def cached_subset_matches(subset_dir: Path, args: argparse.Namespace, target_instruction: str) -> bool:
    metadata_path = get_subset_metadata_path(subset_dir)
    if not metadata_path.exists():
        return False

    metadata = load_subset_metadata(subset_dir)
    if metadata.get("cache_version") != SUBSET_CACHE_VERSION:
        return False
    if metadata.get("dataset_name") != args.dataset_name:
        return False
    if metadata.get("task_instruction") != target_instruction:
        return False
    if metadata.get("center_crop") != args.center_crop:
        return False
    if int(metadata.get("num_cached", 0)) < args.max_samples:
        return False

    samples_dir = get_subset_samples_dir(subset_dir)
    return samples_dir.exists()


def discover_target_instruction(
    vla: nn.Module,
    processor,
    args: argparse.Namespace,
    *,
    label: str = "task",
    exclude_instruction: Optional[str] = None,
) -> str:
    print(f"Discovering {label} instruction from the dataset stream...")
    probe_args = clone_args(args, task_instruction=None)
    dataloader = build_dataloader(vla, processor, probe_args, args.dataset_name)
    tokenizer = processor.tokenizer
    excluded = canonicalize_instruction(exclude_instruction)

    for idx, batch in enumerate(dataloader):
        if idx >= args.max_scan:
            break
        instruction = get_batch_instruction(batch, tokenizer)
        if instruction and instruction != excluded:
            return instruction

    if excluded is None:
        raise RuntimeError(f"Could not discover a {label} instruction within the first {args.max_scan} streamed samples.")
    raise RuntimeError(
        f"Could not discover a {label} instruction different from '{excluded}' within the first {args.max_scan} streamed samples."
    )


def reset_subset_dir(subset_dir: Path) -> None:
    samples_dir = get_subset_samples_dir(subset_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    for sample_path in samples_dir.glob("sample_*.pt"):
        sample_path.unlink()


def prepare_task_subset(
    vla: nn.Module,
    processor,
    args: argparse.Namespace,
    output_dir: Path,
    target_instruction: Optional[str] = None,
    label: str = "task",
    exclude_instruction: Optional[str] = None,
) -> dict:
    if target_instruction is None:
        target_instruction = discover_target_instruction(
            vla, processor, args, label=label, exclude_instruction=exclude_instruction
        )
        print(f"  Auto-selected {label} instruction: '{target_instruction}'")

    subset_dir = get_subset_dir(output_dir, target_instruction)
    if not args.rebuild_subset and cached_subset_matches(subset_dir, args, target_instruction):
        metadata = load_subset_metadata(subset_dir)
        print(f"Reusing cached {label} subset with {metadata['num_cached']} samples for '{metadata['task_instruction']}' from {subset_dir}")
        return metadata

    work_args = clone_args(args, task_instruction=target_instruction)
    dataloader = build_dataloader(vla, processor, work_args, args.dataset_name)
    ds_stats = dataloader.dataset.dataset_statistics[args.dataset_name]
    tokenizer = processor.tokenizer

    reset_subset_dir(subset_dir)
    samples_dir = get_subset_samples_dir(subset_dir)

    print(f"Caching up to {args.max_samples} samples for {label} task '{target_instruction}'...")

    collected = 0
    streamed = 0
    with torch.inference_mode():
        for batch in tqdm(dataloader, total=args.max_samples, desc="caching task subset"):
            streamed += 1
            instruction = get_batch_instruction(batch, tokenizer)
            if instruction != target_instruction:
                continue

            prompt_input_ids, prompt_attention_mask = build_prompt_only_inputs(batch)
            sample = {
                "prompt_input_ids": prompt_input_ids.cpu(),
                "prompt_attention_mask": prompt_attention_mask.cpu(),
                "pixel_values": batch["pixel_values"].cpu(),
                "actions": batch["actions"].cpu(),
            }
            if batch.get("proprio") is not None:
                sample["proprio"] = batch["proprio"].cpu()

            torch.save(sample, samples_dir / f"sample_{collected:04d}.pt")
            collected += 1
            if collected >= args.max_samples:
                break

    if collected == 0:
        raise RuntimeError(f"No samples found for {label} task instruction '{target_instruction}'.")

    metadata = {
        "cache_version": SUBSET_CACHE_VERSION,
        "dataset_name": args.dataset_name,
        "task_instruction": target_instruction,
        "num_cached": collected,
        "num_streamed": streamed,
        "center_crop": args.center_crop,
        "action_stats": json_ready(ds_stats["action"]),
    }
    subset_dir.mkdir(parents=True, exist_ok=True)
    with open(get_subset_metadata_path(subset_dir), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Cached {collected} samples to {subset_dir}")
    return metadata


def iter_cached_subset(output_dir: Path, task_instruction: str, max_samples: int) -> Iterator[Dict[str, torch.Tensor]]:
    subset_dir = get_subset_dir(output_dir, task_instruction)
    samples_dir = get_subset_samples_dir(subset_dir)
    sample_paths = sorted(samples_dir.glob("sample_*.pt"))[:max_samples]
    for sample_path in sample_paths:
        yield torch.load(sample_path, map_location="cpu", weights_only=True)


def collect_same_task_samples(
    vla: nn.Module,
    action_head: nn.Module,
    proprio_projector: nn.Module,
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    output_dir: Path,
    subset_meta: dict,
) -> None:
    ds_dir = output_dir / "collect"
    hiddens_dir = ds_dir / "hiddens"
    hiddens_dir.mkdir(parents=True, exist_ok=True)

    llm_layers, layer_names = resolve_llm_layers(vla)
    layer_indices = list(range(len(llm_layers)))
    num_patches = compute_num_patches(vla, cfg)
    hooks = CaptureHooks(llm_layers, layer_indices)
    action_stats = subset_meta["action_stats"]
    task_instruction = subset_meta["task_instruction"]

    num_samples = min(int(subset_meta["num_cached"]), args.max_samples)
    all_losses: List[float] = []
    all_preds: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []

    print(f"Collecting {num_samples} cached samples for '{task_instruction}'...")

    with torch.inference_mode():
        for sample_idx, sample in enumerate(
            tqdm(iter_cached_subset(output_dir, task_instruction, num_samples), total=num_samples, desc="collecting cached subset")
        ):
            prompt_ids = sample["prompt_input_ids"]
            prompt_mask = sample["prompt_attention_mask"]
            n_prompt = infer_num_prompt_tokens(prompt_ids)

            proprio = None
            if cfg.use_proprio and sample.get("proprio") is not None:
                proprio = sample["proprio"].to(DTYPE)

            hooks.clear()
            pred_raw, _ = vla.predict_action(
                input_ids=prompt_ids.to(DEVICE),
                attention_mask=prompt_mask.to(DEVICE),
                pixel_values=sample["pixel_values"].to(DTYPE).to(DEVICE),
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
                unnorm_key=cfg.unnorm_key,
                use_film=False,
            )

            pred_t = torch.tensor(pred_raw, dtype=torch.float32).unsqueeze(0)
            gt_t = unnormalize_bounds_tensor(sample["actions"], action_stats).to(torch.float32)
            loss = float(torch.abs(pred_t - gt_t).mean().item())

            all_losses.append(loss)
            all_preds.append(np.asarray(pred_raw))
            all_gt.append(gt_t[0].cpu().numpy())

            sample_hiddens = {}
            for li in layer_indices:
                h = extract_predictive_hidden_from_layer(hooks.captured[li], num_patches, n_prompt)
                sample_hiddens[li] = h[0].detach().cpu().to(torch.float16)
            torch.save(sample_hiddens, hiddens_dir / f"sample_{sample_idx:04d}.pt")

    hooks.remove()

    np.save(ds_dir / "predictions.npy", np.stack(all_preds))
    np.save(ds_dir / "ground_truth.npy", np.stack(all_gt))

    with open(ds_dir / "losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "loss"])
        for idx, loss in enumerate(all_losses):
            writer.writerow([idx, loss])

    metadata = {
        "dataset_name": args.dataset_name,
        "task_instruction": task_instruction,
        "num_collected": num_samples,
        "num_layers": len(layer_indices),
        "layer_names": layer_names,
        "mean_loss": float(np.mean(all_losses)),
        "std_loss": float(np.std(all_losses)),
        "subset_dir": str(get_subset_dir(output_dir, task_instruction)),
        "subset_num_cached": int(subset_meta["num_cached"]),
    }
    with open(ds_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Collected {num_samples} cached samples, mean loss={np.mean(all_losses):.4f}")


def compute_vectors_from_disk(output_dir: Path) -> None:
    ds_dir = output_dir / "collect"
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    with open(ds_dir / "metadata.json") as f:
        metadata = json.load(f)
    num_layers = metadata["num_layers"]
    layer_indices = list(range(num_layers))

    losses: List[float] = []
    with open(ds_dir / "losses.csv") as f:
        for row in csv.DictReader(f):
            losses.append(float(row["loss"]))
    losses_t = torch.tensor(losses)

    sorted_idx = torch.argsort(losses_t)
    q = max(1, len(sorted_idx) // 4)
    good_set = sorted_idx[:q].tolist()
    bad_set = sorted_idx[-q:].tolist()

    print(
        f"  {q} good samples (loss <= {losses_t[sorted_idx[q - 1]]:.4f}), "
        f"{q} bad samples (loss >= {losses_t[sorted_idx[-q]]:.4f}), "
        f"middle {len(losses) - 2 * q} ignored, total {len(losses)}"
    )

    hiddens_dir = ds_dir / "hiddens"
    vectors: Dict[int, torch.Tensor] = {}
    raw_norms: Dict[int, float] = {}

    for li in tqdm(layer_indices, desc="computing vectors"):
        good_h = []
        for si in good_set:
            sample_hiddens = torch.load(hiddens_dir / f"sample_{si:04d}.pt", map_location="cpu", weights_only=True)
            good_h.append(sample_hiddens[li].to(torch.float32).reshape(-1))
        good_mean = torch.stack(good_h).mean(0)

        bad_h = []
        for si in bad_set:
            sample_hiddens = torch.load(hiddens_dir / f"sample_{si:04d}.pt", map_location="cpu", weights_only=True)
            bad_h.append(sample_hiddens[li].to(torch.float32).reshape(-1))
        bad_mean = torch.stack(bad_h).mean(0)

        vector = good_mean - bad_mean
        raw_norms[li] = float(vector.norm().item())
        vectors[li] = vector.reshape(ACTION_DIM * NUM_ACTIONS_CHUNK, -1)
        print(f"    L{li}: raw_norm={raw_norms[li]:.2f}")

    torch.save(vectors, vectors_dir / "steering_vectors.pt")

    vector_meta = {
        "quartile_size": q,
        "num_samples": len(losses),
        "good_loss_threshold": float(losses_t[sorted_idx[q - 1]]),
        "bad_loss_threshold": float(losses_t[sorted_idx[-q]]),
        "raw_norms": {str(k): v for k, v in raw_norms.items()},
    }
    with open(vectors_dir / "vector_metadata.json", "w") as f:
        json.dump(vector_meta, f, indent=2)

    print(f"  Saved steering vectors to {vectors_dir / 'steering_vectors.pt'}")


def compute_spearman_from_disk(output_dir: Path) -> None:
    ds_dir = output_dir / "collect"
    vectors_dir = output_dir / "vectors"
    spearman_dir = output_dir / "spearman"
    spearman_dir.mkdir(parents=True, exist_ok=True)

    with open(ds_dir / "metadata.json") as f:
        metadata = json.load(f)
    num_layers = metadata["num_layers"]
    layer_indices = list(range(num_layers))

    losses: List[float] = []
    with open(ds_dir / "losses.csv") as f:
        for row in csv.DictReader(f):
            losses.append(float(row["loss"]))
    n_samples = len(losses)

    vectors = torch.load(vectors_dir / "steering_vectors.pt", map_location="cpu", weights_only=True)

    per_layer_cosines: Dict[int, List[float]] = {li: [] for li in layer_indices}
    avg_cosines: List[float] = []

    for si in tqdm(range(n_samples), desc="computing cosine similarities"):
        sample_h = torch.load(ds_dir / "hiddens" / f"sample_{si:04d}.pt", map_location="cpu", weights_only=True)
        layer_cosines = []
        for li in layer_indices:
            hidden = sample_h[li].to(torch.float32).reshape(-1)
            vector = vectors[li].reshape(-1)
            cosine = float(torch.nn.functional.cosine_similarity(hidden.unsqueeze(0), vector.unsqueeze(0)).item())
            per_layer_cosines[li].append(cosine)
            layer_cosines.append(cosine)
        avg_cosines.append(float(np.mean(layer_cosines)))

    results = {"per_layer": {}, "average": {}}

    rho, pval = spearmanr(losses, avg_cosines)
    results["average"] = {
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "n_samples": n_samples,
    }
    print(f"  Average across layers: rho={rho:.4f}, p={pval:.4e}")

    for li in layer_indices:
        rho, pval = spearmanr(losses, per_layer_cosines[li])
        results["per_layer"][str(li)] = {
            "spearman_rho": float(rho),
            "p_value": float(pval),
        }

    per_sample = {"sample_index": list(range(n_samples)), "loss": losses, "avg_cosine": avg_cosines}
    for li in layer_indices:
        per_sample[f"cosine_layer_{li}"] = per_layer_cosines[li]

    with open(spearman_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.writer(f)
        headers = list(per_sample.keys())
        writer.writerow(headers)
        for i in range(n_samples):
            writer.writerow([per_sample[key][i] for key in headers])

    with open(spearman_dir / "correlations.json", "w") as f:
        json.dump(results, f, indent=2)

    sorted_layers = sorted(results["per_layer"].items(), key=lambda item: item[1]["spearman_rho"])
    print("  Top 5 layers (most negative rho = steering aligns with lower loss):")
    for layer_name, layer_result in sorted_layers[:5]:
        print(f"    L{layer_name}: rho={layer_result['spearman_rho']:.4f}")
    print("  Bottom 5 layers (most positive rho):")
    for layer_name, layer_result in sorted_layers[-5:]:
        print(f"    L{layer_name}: rho={layer_result['spearman_rho']:.4f}")


def load_baseline_losses(output_dir: Path, num_samples: int) -> np.ndarray:
    losses: List[float] = []
    with open(output_dir / "collect" / "losses.csv") as f:
        for row in csv.DictReader(f):
            losses.append(float(row["loss"]))

    if len(losses) < num_samples:
        raise RuntimeError(
            f"Collect phase only has {len(losses)} baseline losses, but steer requested {num_samples} samples."
        )
    return np.asarray(losses[:num_samples], dtype=np.float32)


def try_load_collect_baseline(output_dir: Path, task_instruction: str, num_samples: int) -> Optional[np.ndarray]:
    collect_metadata_path = output_dir / "collect" / "metadata.json"
    if not collect_metadata_path.exists():
        return None

    with open(collect_metadata_path) as f:
        metadata = json.load(f)

    if metadata.get("task_instruction") != task_instruction:
        return None
    if int(metadata.get("num_collected", 0)) < num_samples:
        return None
    return load_baseline_losses(output_dir, num_samples)


def steered_inference(
    vla: nn.Module,
    action_head: nn.Module,
    proprio_projector: nn.Module,
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    output_dir: Path,
    subset_meta: dict,
    vector_task_instruction: str,
) -> None:
    steer_dir = output_dir / "steered" / f"c{str(args.coeff).replace('.', 'p')}"
    steer_dir.mkdir(parents=True, exist_ok=True)

    vectors = torch.load(output_dir / "vectors" / "steering_vectors.pt", map_location="cpu", weights_only=True)
    llm_layers, _ = resolve_llm_layers(vla)
    layer_indices = list(range(len(llm_layers)))
    num_patches = compute_num_patches(vla, cfg)
    action_stats = subset_meta["action_stats"]
    eval_task_instruction = subset_meta["task_instruction"]
    num_samples = min(int(subset_meta["num_cached"]), args.max_samples)
    baseline_losses_np = try_load_collect_baseline(output_dir, eval_task_instruction, num_samples)
    reuse_collect_baseline = baseline_losses_np is not None

    steering = MultiLayerSteeringHooks(
        llm_layers,
        layer_indices,
        vectors,
        args.coeff,
        num_patches,
        preserve_norm=args.preserve_norm,
    )

    recomputed_baseline_losses: List[float] = []
    steered_losses: List[float] = []
    steered_preds: List[np.ndarray] = []

    if reuse_collect_baseline:
        print(
            f"Running steered inference (coeff={args.coeff}) on {num_samples} cached eval samples of "
            f"'{eval_task_instruction}' using collect baselines from the same task..."
        )
    else:
        print(
            f"Running steered inference (coeff={args.coeff}) on {num_samples} cached eval samples of "
            f"'{eval_task_instruction}' with an in-loop baseline recomputation because the vector task differs..."
        )

    try:
        with torch.inference_mode():
            for sample in tqdm(
                iter_cached_subset(output_dir, eval_task_instruction, num_samples),
                total=num_samples,
                desc="steered pass",
            ):
                prompt_ids = sample["prompt_input_ids"]
                prompt_mask = sample["prompt_attention_mask"]
                n_prompt = infer_num_prompt_tokens(prompt_ids)

                proprio = None
                if cfg.use_proprio and sample.get("proprio") is not None:
                    proprio = sample["proprio"].to(DTYPE)

                predict_kwargs = dict(
                    input_ids=prompt_ids.to(DEVICE),
                    attention_mask=prompt_mask.to(DEVICE),
                    pixel_values=sample["pixel_values"].to(DTYPE).to(DEVICE),
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                    unnorm_key=cfg.unnorm_key,
                    use_film=False,
                )
                gt_t = unnormalize_bounds_tensor(sample["actions"], action_stats).to(torch.float32)

                if not reuse_collect_baseline:
                    steering.set_num_prompt_tokens(None)
                    baseline_pred, _ = vla.predict_action(**predict_kwargs)
                    baseline_t = torch.tensor(baseline_pred, dtype=torch.float32).unsqueeze(0)
                    recomputed_baseline_losses.append(float(torch.abs(baseline_t - gt_t).mean().item()))

                steering.set_num_prompt_tokens(n_prompt)
                steered_pred, _ = vla.predict_action(**predict_kwargs)
                steering.set_num_prompt_tokens(None)

                steered_t = torch.tensor(steered_pred, dtype=torch.float32).unsqueeze(0)
                steered_losses.append(float(torch.abs(steered_t - gt_t).mean().item()))
                steered_preds.append(np.asarray(steered_pred))
    finally:
        steering.remove()

    if not reuse_collect_baseline:
        baseline_losses_np = np.asarray(recomputed_baseline_losses, dtype=np.float32)
    steered_losses_np = np.array(steered_losses)
    deltas = steered_losses_np - baseline_losses_np

    np.save(steer_dir / "steered_predictions.npy", np.stack(steered_preds))

    with open(steer_dir / "per_sample.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "baseline_loss", "steered_loss", "loss_delta"])
        for idx in range(len(steered_losses_np)):
            writer.writerow([idx, baseline_losses_np[idx], steered_losses_np[idx], deltas[idx]])

    summary = {
        "vector_task_instruction": vector_task_instruction,
        "eval_task_instruction": eval_task_instruction,
        "dataset_name": args.dataset_name,
        "model": args.pretrained_checkpoint,
        "coeff": args.coeff,
        "preserve_norm": args.preserve_norm,
        "num_samples": len(steered_losses_np),
        "baseline_source": "collect" if reuse_collect_baseline else "recomputed_eval_task",
        "baseline_avg_loss": float(baseline_losses_np.mean()),
        "steered_avg_loss": float(steered_losses_np.mean()),
        "avg_loss_delta": float(deltas.mean()),
        "relative_change": float(deltas.mean() / baseline_losses_np.mean()) if baseline_losses_np.mean() != 0 else 0.0,
        "improved_fraction": float((deltas < 0).mean()),
    }
    with open(steer_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cfg = make_cfg(args)
    vector_task_instruction, eval_task_instruction = resolve_task_instructions(args)

    run_prepare = args.phase in ("all", "prepare", "collect", "steer")
    run_collect = args.phase in ("all", "collect")
    run_vectors = args.phase in ("all", "vectors")
    run_spearman = args.phase in ("all", "spearman")
    run_steer = args.phase in ("all", "steer")

    needs_vla = run_prepare or run_collect or run_steer
    needs_action_modules = run_collect or run_steer

    vector_subset_meta = None
    eval_subset_meta = None
    if needs_vla:
        vla = get_vla(cfg)
        processor = get_processor(cfg)

    if run_prepare:
        print("=" * 60)
        print("Phase 0: Prepare cached task subsets")
        print("=" * 60)
        vector_subset_meta = prepare_task_subset(
            vla, processor, args, output_dir, target_instruction=vector_task_instruction, label="vector"
        )
        if eval_task_instruction is None:
            eval_subset_meta = prepare_task_subset(
                vla,
                processor,
                args,
                output_dir,
                target_instruction=None,
                label="eval",
                exclude_instruction=vector_subset_meta["task_instruction"],
            )
            eval_task_instruction = eval_subset_meta["task_instruction"]
        if eval_subset_meta is None and eval_task_instruction == vector_subset_meta["task_instruction"]:
            eval_subset_meta = vector_subset_meta
        elif eval_subset_meta is None:
            eval_subset_meta = prepare_task_subset(
                vla, processor, args, output_dir, target_instruction=eval_task_instruction, label="eval"
            )

    if needs_action_modules:
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    if run_collect:
        if vector_subset_meta is None:
            if vector_task_instruction is None:
                raise RuntimeError("Collect phase requires a vector task instruction or a prepared vector subset.")
            vector_subset_meta = load_subset_metadata(get_subset_dir(output_dir, vector_task_instruction))

        print("=" * 60)
        print("Phase 1: Collect residual streams from cached task samples")
        print("=" * 60)
        collect_same_task_samples(vla, action_head, proprio_projector, cfg, args, output_dir, vector_subset_meta)

    if run_vectors:
        print("=" * 60)
        print("Phase 2: Compute steering vectors from stored hidden states")
        print("=" * 60)
        compute_vectors_from_disk(output_dir)

    if run_spearman:
        print("=" * 60)
        print("Phase 3: Spearman correlation (cosine similarity vs loss)")
        print("=" * 60)
        compute_spearman_from_disk(output_dir)

    if run_steer:
        if eval_subset_meta is None:
            if eval_task_instruction is None:
                raise RuntimeError("Steer phase requires an eval task instruction or a prepared eval subset.")
            eval_subset_meta = load_subset_metadata(get_subset_dir(output_dir, eval_task_instruction))

        print("=" * 60)
        print("Phase 4: Steered inference")
        print("=" * 60)
        steered_inference(
            vla,
            action_head,
            proprio_projector,
            cfg,
            args,
            output_dir,
            eval_subset_meta,
            vector_task_instruction=vector_subset_meta["task_instruction"] if vector_subset_meta is not None else (vector_task_instruction or eval_task_instruction),
        )


if __name__ == "__main__":
    main()
