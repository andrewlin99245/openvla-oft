"""Full steering evaluation pipeline for the general (all-four-tasks) OpenVLA-OFT model.

Phase 1 – Collect:  For each of 3 LIBERO datasets, run inference sample-by-sample,
                    storing per-layer residual streams, predicted actions, and losses.
Phase 2 – Vectors:  Compute steering vectors from stored hidden states
                    (good-quartile mean minus bad-quartile mean).
Phase 3 – Steer:   For every (vector_source, eval_target) pair (3x3 = 9 runs),
                    run steered inference and save predicted actions and losses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from experiments.analysis.libero_sira_correlation import (
    ACTION_DIM,
    DEVICE,
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

DATASETS = [
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
]
DEFAULT_MODEL = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full steering eval: collect -> vectors -> steer (3x3).")
    p.add_argument("--data-root-dir", required=True)
    p.add_argument("--output-dir", default="outputs/full_steering_eval")
    p.add_argument("--pretrained-checkpoint", default=DEFAULT_MODEL)
    p.add_argument("--max-samples", type=int, default=512)
    p.add_argument("--coeff", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--phase", choices=["all", "collect", "vectors", "spearman", "steer"], default="all",
                   help="Run only a specific phase (default: all)")
    p.add_argument("--preserve-norm", action="store_true", default=False,
                   help="Rescale steered hidden states to preserve original L2 norm (default: off)")
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
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        lora_rank=32,
        unnorm_key="",
        load_in_8bit=False,
        load_in_4bit=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Collect residual streams, predictions, and losses
# ──────────────────────────────────────────────────────────────────────────────

def collect_dataset(
    dataset_name: str,
    vla: nn.Module,
    processor,
    action_head: nn.Module,
    proprio_projector: nn.Module,
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    ds_dir = output_dir / "collect" / dataset_name
    hiddens_dir = ds_dir / "hiddens"
    hiddens_dir.mkdir(parents=True, exist_ok=True)

    dataloader = build_dataloader(vla, processor, args, dataset_name)
    ds_stats = dataloader.dataset.dataset_statistics[dataset_name]

    llm_layers, _ = resolve_llm_layers(vla)
    layer_indices = list(range(len(llm_layers)))
    num_patches = compute_num_patches(vla, cfg)

    hooks = CaptureHooks(llm_layers, layer_indices)

    all_losses: List[float] = []
    all_preds: List[np.ndarray] = []
    all_gt: List[np.ndarray] = []

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dataloader, total=args.max_samples, desc=f"collect {dataset_name}")):
            if i >= args.max_samples:
                break

            prompt_ids, prompt_mask = build_prompt_only_inputs(batch)
            n_prompt = infer_num_prompt_tokens(prompt_ids)

            proprio = None
            if cfg.use_proprio and batch.get("proprio") is not None:
                proprio = batch["proprio"].to(torch.bfloat16)

            hooks.clear()
            pred_raw, _ = vla.predict_action(
                input_ids=prompt_ids.to(DEVICE),
                attention_mask=prompt_mask.to(DEVICE),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(DEVICE),
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
                unnorm_key=dataset_name,
                use_film=False,
            )

            pred_t = torch.tensor(pred_raw, dtype=torch.float32).unsqueeze(0)
            gt_t = unnormalize_bounds_tensor(batch["actions"], ds_stats["action"]).to(torch.float32)
            loss = float(torch.abs(pred_t - gt_t).mean().item())

            all_losses.append(loss)
            all_preds.append(np.asarray(pred_raw))
            all_gt.append(gt_t[0].cpu().numpy())

            # Save per-layer hidden states to disk one sample at a time
            sample_hiddens = {}
            for li in layer_indices:
                h = extract_predictive_hidden_from_layer(hooks.captured[li], num_patches, n_prompt)
                sample_hiddens[li] = h[0].detach().cpu().to(torch.float16)
            torch.save(sample_hiddens, hiddens_dir / f"sample_{i:04d}.pt")

            del sample_hiddens
            torch.cuda.empty_cache()

    hooks.remove()

    # Save predictions, ground truth, and losses
    np.save(ds_dir / "predictions.npy", np.stack(all_preds))
    np.save(ds_dir / "ground_truth.npy", np.stack(all_gt))
    with open(ds_dir / "losses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_index", "loss"])
        for idx, l in enumerate(all_losses):
            w.writerow([idx, l])

    print(f"  Collected {len(all_losses)} samples for {dataset_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Compute steering vectors from stored hidden states
# ──────────────────────────────────────────────────────────────────────────────

def compute_vectors_from_disk(output_dir: Path, num_layers: int) -> None:
    vectors_dir = output_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = list(range(num_layers))

    for dataset_name in DATASETS:
        ds_dir = output_dir / "collect" / dataset_name
        hiddens_dir = ds_dir / "hiddens"

        # Load losses
        losses: List[float] = []
        with open(ds_dir / "losses.csv") as f:
            for row in csv.DictReader(f):
                losses.append(float(row["loss"]))
        losses_t = torch.tensor(losses)

        # Quartile split
        sorted_idx = torch.argsort(losses_t)
        q = max(1, len(sorted_idx) // 4)
        good_set = sorted_idx[:q].tolist()
        bad_set = sorted_idx[-q:].tolist()

        print(f"  {dataset_name}: {q} good, {q} bad samples "
              f"(middle {len(losses) - 2 * q} ignored) from {len(losses)} total")

        # Compute vectors one layer at a time to limit memory
        vectors: Dict[int, torch.Tensor] = {}
        for li in tqdm(layer_indices, desc=f"vectors {dataset_name}"):
            good_h = []
            for si in good_set:
                d = torch.load(hiddens_dir / f"sample_{si:04d}.pt", map_location="cpu", weights_only=True)
                good_h.append(d[li].to(torch.float32).reshape(-1))
            good_mean = torch.stack(good_h).mean(0)
            del good_h

            bad_h = []
            for si in bad_set:
                d = torch.load(hiddens_dir / f"sample_{si:04d}.pt", map_location="cpu", weights_only=True)
                bad_h.append(d[li].to(torch.float32).reshape(-1))
            bad_mean = torch.stack(bad_h).mean(0)
            del bad_h

            v = good_mean - bad_mean
            print(f"    L{li}: raw_norm={v.norm().item():.2f}")
            vectors[li] = v.reshape(ACTION_DIM * NUM_ACTIONS_CHUNK, -1)

        torch.save(vectors, vectors_dir / f"{dataset_name}.pt")
        print(f"  Saved steering vectors for {dataset_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2.5: Spearman correlation (cosine similarity vs loss)
# ──────────────────────────────────────────────────────────────────────────────

def compute_spearman_correlations(output_dir: Path, num_layers: int) -> None:
    """For each (vector_source, eval_dataset) pair, compute per-sample cosine
    similarity between the steering vector and the sample's residual stream,
    then Spearman correlation with loss.  CPU-only, no model needed."""
    from scipy.stats import spearmanr

    vectors_dir = output_dir / "vectors"
    spearman_dir = output_dir / "spearman"
    spearman_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = list(range(num_layers))

    # Load all vectors upfront
    all_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
    for ds in DATASETS:
        all_vectors[ds] = torch.load(vectors_dir / f"{ds}.pt", map_location="cpu", weights_only=True)

    results = {}
    for eval_ds in DATASETS:
        ds_dir = output_dir / "collect" / eval_ds
        hiddens_dir = ds_dir / "hiddens"

        # Load losses
        losses: List[float] = []
        with open(ds_dir / "losses.csv") as f:
            for row in csv.DictReader(f):
                losses.append(float(row["loss"]))

        n_samples = len(losses)

        # Per-sample cosine similarities for each vector source
        # {vec_ds: [per-sample avg cosine across layers]}
        cosines: Dict[str, List[float]] = {ds: [] for ds in DATASETS}

        for si in tqdm(range(n_samples), desc=f"cosine for eval={eval_ds}"):
            sample_h = torch.load(hiddens_dir / f"sample_{si:04d}.pt", map_location="cpu", weights_only=True)
            for vec_ds in DATASETS:
                vectors = all_vectors[vec_ds]
                layer_cos = []
                for li in layer_indices:
                    h = sample_h[li].to(torch.float32).reshape(-1)
                    v = vectors[li].reshape(-1)
                    cos = float(torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item())
                    layer_cos.append(cos)
                cosines[vec_ds].append(float(np.mean(layer_cos)))

        for vec_ds in DATASETS:
            rho, pval = spearmanr(losses, cosines[vec_ds])
            tag = f"{vec_ds}_to_{eval_ds}"
            results[tag] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_samples": n_samples,
            }
            print(f"  {tag}: rho={rho:.4f}, p={pval:.4e}, n={n_samples}")

    with open(spearman_dir / "correlations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Spearman correlations (cosine similarity vs loss) ===")
    print(json.dumps(results, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Steered inference (3 baselines + 9 steered = 12 runs)
# ──────────────────────────────────────────────────────────────────────────────

def steer_eval_dataset(
    eval_dataset: str,
    all_vectors: Dict[str, Dict[int, torch.Tensor]],
    vla: nn.Module,
    processor,
    action_head: nn.Module,
    proprio_projector: nn.Module,
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Run 1 baseline + N steered passes per sample for one eval dataset.

    all_vectors maps vector_dataset_name -> {layer_idx: tensor}.
    Baseline is computed once and shared across all vector sources.
    """
    coeff_tag = f"c{str(args.coeff).replace('.', 'p')}"
    vec_ds_names = list(all_vectors.keys())

    llm_layers, _ = resolve_llm_layers(vla)
    layer_indices = list(range(len(llm_layers)))
    num_patches = compute_num_patches(vla, cfg)

    dataloader = build_dataloader(vla, processor, args, eval_dataset)
    ds_stats = dataloader.dataset.dataset_statistics[eval_dataset]

    # Single hooks object — swap vectors via set_vectors()
    steering = MultiLayerSteeringHooks(
        llm_layers, layer_indices, all_vectors[vec_ds_names[0]],
        args.coeff, num_patches, preserve_norm=args.preserve_norm,
    )

    baseline_losses: List[float] = []
    steered_losses: Dict[str, List[float]] = {ds: [] for ds in vec_ds_names}
    steered_preds: Dict[str, List[np.ndarray]] = {ds: [] for ds in vec_ds_names}

    n_forward = 1 + len(vec_ds_names)  # per sample
    try:
        with torch.inference_mode():
            for i, batch in enumerate(tqdm(
                dataloader, total=args.max_samples,
                desc=f"eval {eval_dataset} ({n_forward} fwd/sample)",
            )):
                if i >= args.max_samples:
                    break

                prompt_ids, prompt_mask = build_prompt_only_inputs(batch)
                n_prompt = infer_num_prompt_tokens(prompt_ids)

                proprio = None
                if cfg.use_proprio and batch.get("proprio") is not None:
                    proprio = batch["proprio"].to(torch.bfloat16)

                predict_kwargs = dict(
                    input_ids=prompt_ids.to(DEVICE),
                    attention_mask=prompt_mask.to(DEVICE),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(DEVICE),
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    action_head=action_head,
                    unnorm_key=eval_dataset,
                    use_film=False,
                )
                gt_t = unnormalize_bounds_tensor(batch["actions"], ds_stats["action"]).to(torch.float32)

                # Baseline (no steering) — once per sample
                steering.set_num_prompt_tokens(None)
                bl_pred, _ = vla.predict_action(**predict_kwargs)
                bl_t = torch.tensor(bl_pred, dtype=torch.float32).unsqueeze(0)
                baseline_losses.append(float(torch.abs(bl_t - gt_t).mean().item()))

                # Steered — once per vector source
                for vec_ds in vec_ds_names:
                    steering.set_vectors(all_vectors[vec_ds])
                    steering.set_num_prompt_tokens(n_prompt)
                    pred_raw, _ = vla.predict_action(**predict_kwargs)
                    steering.set_num_prompt_tokens(None)

                    pred_t = torch.tensor(pred_raw, dtype=torch.float32).unsqueeze(0)
                    steered_losses[vec_ds].append(float(torch.abs(pred_t - gt_t).mean().item()))
                    steered_preds[vec_ds].append(np.asarray(pred_raw))
    finally:
        steering.remove()

    # Save results for each (vec_ds, eval_dataset) pair
    bl = np.array(baseline_losses)
    for vec_ds in vec_ds_names:
        tag = f"{vec_ds}_to_{eval_dataset}"
        run_dir = output_dir / "steered" / coeff_tag / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        sl = np.array(steered_losses[vec_ds])
        deltas = sl - bl

        np.save(run_dir / "predictions.npy", np.stack(steered_preds[vec_ds]))

        with open(run_dir / "per_sample.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_index", "baseline_loss", "steered_loss", "loss_delta"])
            for idx in range(len(sl)):
                w.writerow([idx, bl[idx], sl[idx], deltas[idx]])

        summary = {
            "vector_dataset": vec_ds,
            "eval_dataset": eval_dataset,
            "model": args.pretrained_checkpoint,
            "coeff": args.coeff,
            "preserve_norm": args.preserve_norm,
            "num_layers": len(layer_indices),
            "num_samples": len(sl),
            "baseline_avg_loss": float(bl.mean()),
            "steered_avg_loss": float(sl.mean()),
            "avg_loss_delta": float(deltas.mean()),
            "relative_change": float(deltas.mean() / bl.mean()) if bl.mean() != 0 else 0.0,
            "improved_fraction": float((deltas < 0).mean()),
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    cfg = make_cfg(args)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    llm_layers, _ = resolve_llm_layers(vla)
    num_layers = len(llm_layers)

    run_collect = args.phase in ("all", "collect")
    run_vectors = args.phase in ("all", "vectors")
    run_spearman = args.phase in ("all", "spearman")
    run_steer = args.phase in ("all", "steer")

    # Phase 1
    if run_collect:
        print("=" * 60)
        print("Phase 1: Collect residual streams, predictions, losses")
        print("=" * 60)
        for ds in DATASETS:
            collect_dataset(ds, vla, processor, action_head, proprio_projector, cfg, args, output_dir)

    # Phase 2
    if run_vectors:
        print("=" * 60)
        print("Phase 2: Compute steering vectors from stored hidden states")
        print("=" * 60)
        compute_vectors_from_disk(output_dir, num_layers)

    # Phase 2.5: Spearman correlations (CPU-only)
    if run_spearman:
        print("=" * 60)
        print("Phase 2.5: Spearman correlations (cosine similarity vs loss)")
        print("=" * 60)
        compute_spearman_correlations(output_dir, num_layers)

    # Phase 3
    if run_steer:
        print("=" * 60)
        print("Phase 3: Steered inference (3 baselines + 9 steered = 12 runs)")
        print("=" * 60)
        # Load all vectors upfront
        all_vectors = {}
        for ds in DATASETS:
            all_vectors[ds] = torch.load(
                output_dir / "vectors" / f"{ds}.pt", map_location="cpu", weights_only=True,
            )
        # Iterate per eval dataset: 1 baseline + 3 steered per sample
        for eval_ds in DATASETS:
            steer_eval_dataset(
                eval_ds, all_vectors, vla, processor, action_head,
                proprio_projector, cfg, args, output_dir,
            )


if __name__ == "__main__":
    main()
