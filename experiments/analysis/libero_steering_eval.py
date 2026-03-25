from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from experiments.analysis.libero_sira_correlation import (
    ACTION_DIM,
    DEVICE,
    DTYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    adapt_batch_normalization,
    build_cfg,
    build_dataloader,
    build_prompt_only_inputs,
    compute_num_patches,
    extract_predictive_hidden_from_layer,
    flatten_hidden,
    infer_layer_indices,
    infer_num_prompt_tokens,
    resolve_llm_layers,
    unnormalize_bounds_tensor,
)
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla


class CaptureHooks:
    def __init__(self, layers: Sequence[nn.Module], layer_indices: Sequence[int]) -> None:
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles = []
        for idx in layer_indices:
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def _make_hook(self, idx: int):
        def hook_fn(module, inputs, output):
            self.captured[idx] = output[0] if isinstance(output, tuple) else output
        return hook_fn

    def clear(self) -> None:
        self.captured.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.captured.clear()


class MultiLayerSteeringHooks:
    def __init__(
        self,
        layers: Sequence[nn.Module],
        layer_indices: Sequence[int],
        vector_tokens: Dict[int, torch.Tensor],
        coeff: float,
        num_patches: int,
        preserve_norm: bool = False,
    ) -> None:
        self.vector_tokens = vector_tokens
        self.coeff = coeff
        self.num_patches = num_patches
        self.preserve_norm = preserve_norm
        self.num_prompt_tokens: int | None = None
        self._handles = []
        for idx in layer_indices:
            self._handles.append(layers[idx].register_forward_hook(self._make_hook(idx)))

    def set_num_prompt_tokens(self, num_prompt_tokens: int | None) -> None:
        self.num_prompt_tokens = num_prompt_tokens

    def set_vectors(self, vector_tokens: Dict[int, torch.Tensor]) -> None:
        self.vector_tokens = vector_tokens

    def _make_hook(self, idx: int):
        def hook_fn(module, inputs, output):
            if self.num_prompt_tokens is None:
                return output

            hidden = output[0] if isinstance(output, tuple) else output
            layer_vector = self.vector_tokens[idx]
            start = self.num_patches + self.num_prompt_tokens
            end = start + layer_vector.shape[0]

            steered_hidden = hidden.clone()
            delta = (self.coeff * layer_vector).to(device=hidden.device, dtype=hidden.dtype)
            steered_hidden[:, start:end, :] = steered_hidden[:, start:end, :] + delta.unsqueeze(0)

            if self.preserve_norm:
                original_norm = hidden[:, start:end, :].norm(dim=-1, keepdim=True)
                steered_norm = steered_hidden[:, start:end, :].norm(dim=-1, keepdim=True)
                steered_hidden[:, start:end, :] = steered_hidden[:, start:end, :] * (original_norm / (steered_norm + 1e-12))

            if isinstance(output, tuple):
                return (steered_hidden, *output[1:])
            return steered_hidden

        return hook_fn

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject SIRA vectors into OpenVLA predictive hidden states.")
    parser.add_argument("--data-root-dir", required=True)
    parser.add_argument("--vector-dataset-name", default="libero_object_no_noops")
    parser.add_argument("--eval-dataset-name", default="libero_goal_no_noops")
    parser.add_argument(
        "--pretrained-checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
    )
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    parser.add_argument("--coeff", type=float, default=0.1)
    parser.add_argument(
        "--normalize-vector",
        action="store_true",
        help="Unit-normalize each steering vector before injection. Disabled by default for intervention.",
    )
    parser.add_argument("--output-dir", default="outputs/libero_object_to_goal_with_spatial_steering")
    parser.add_argument("--center-crop", action="store_true", default=True)
    return parser.parse_args()


def predict_loss(
    raw_batch: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg,
    vla,
    action_head,
    proprio_projector,
) -> float:
    prompt_input_ids, prompt_attention_mask = build_prompt_only_inputs(batch)
    proprio = None
    if cfg.use_proprio and batch.get("proprio") is not None:
        proprio = batch["proprio"].to(DTYPE)

    predicted_actions_raw, _ = vla.predict_action(
        input_ids=prompt_input_ids.to(DEVICE),
        attention_mask=prompt_attention_mask.to(DEVICE),
        pixel_values=batch["pixel_values"].to(DTYPE).to(DEVICE),
        proprio=proprio,
        proprio_projector=proprio_projector,
        action_head=action_head,
        unnorm_key=cfg.unnorm_key,
        use_film=False,
    )
    predicted_actions = torch.tensor(predicted_actions_raw, dtype=torch.float32).unsqueeze(0)
    ground_truth_actions = unnormalize_bounds_tensor(raw_batch["actions"], raw_batch["_current_action_stats"]).to(torch.float32)
    return float(torch.abs(predicted_actions - ground_truth_actions).mean(dim=(1, 2)).item())


def compute_vectors(
    dataloader,
    cfg,
    vla,
    action_head,
    proprio_projector,
    llm_layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    num_patches: int,
    max_samples: int,
    current_stats: Dict[str, Dict[str, Sequence[float]]],
    target_stats: Dict[str, Dict[str, Sequence[float]]],
    normalize_vector: bool,
) -> tuple[Dict[int, torch.Tensor], Dict[int, float], int, int]:
    hooks = CaptureHooks(llm_layers, layer_indices)
    all_hiddens = {idx: [] for idx in layer_indices}
    all_losses = []
    seen = 0
    total_batches = max(1, (max_samples + dataloader.batch_size - 1) // dataloader.batch_size)

    with torch.inference_mode():
        for raw_batch in tqdm(dataloader, total=total_batches, desc="vector-pass"):
            raw_batch = dict(raw_batch)
            raw_batch["_current_action_stats"] = current_stats["action"]
            batch = adapt_batch_normalization(raw_batch, current_stats, target_stats)
            prompt_input_ids, prompt_attention_mask = build_prompt_only_inputs(batch)
            num_prompt_tokens = infer_num_prompt_tokens(prompt_input_ids)

            proprio = None
            if cfg.use_proprio and batch.get("proprio") is not None:
                proprio = batch["proprio"].to(DTYPE)

            hooks.clear()
            predicted_actions_raw, _ = vla.predict_action(
                input_ids=prompt_input_ids.to(DEVICE),
                attention_mask=prompt_attention_mask.to(DEVICE),
                pixel_values=batch["pixel_values"].to(DTYPE).to(DEVICE),
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
                unnorm_key=cfg.unnorm_key,
                use_film=False,
            )
            predicted_actions = torch.tensor(predicted_actions_raw, dtype=torch.float32).unsqueeze(0)
            ground_truth_actions = unnormalize_bounds_tensor(raw_batch["actions"], current_stats["action"]).to(torch.float32)
            all_losses.append(torch.abs(predicted_actions - ground_truth_actions).mean(dim=(1, 2)).cpu())

            for idx in layer_indices:
                actions_hidden = extract_predictive_hidden_from_layer(hooks.captured[idx], num_patches, num_prompt_tokens)
                all_hiddens[idx].append(flatten_hidden(actions_hidden.detach().cpu()).to(torch.float16))

            seen += raw_batch["input_ids"].shape[0]
            if seen >= max_samples:
                break

    hooks.remove()

    losses = torch.cat(all_losses, dim=0)[:max_samples]
    sorted_indices = torch.argsort(losses)
    quartile = max(1, len(sorted_indices) // 4)
    good_indices = sorted_indices[:quartile]
    bad_indices = sorted_indices[-quartile:]

    vectors = {}
    raw_norms = {}
    for idx in layer_indices:
        h = torch.cat(all_hiddens[idx], dim=0)[:max_samples].to(torch.float32)
        vector = h[good_indices].mean(dim=0) - h[bad_indices].mean(dim=0)
        raw_norm = float(vector.norm().item())
        raw_norms[idx] = raw_norm
        if normalize_vector:
            vector = vector / (vector.norm() + 1e-12)
        vectors[idx] = vector.reshape(ACTION_DIM * NUM_ACTIONS_CHUNK, -1).to(torch.float32)

    return vectors, raw_norms, quartile, int(len(losses))


def evaluate_baseline_and_steered(
    dataloader,
    cfg,
    vla,
    action_head,
    proprio_projector,
    llm_layers: Sequence[nn.Module],
    layer_indices: Sequence[int],
    vector_tokens: Dict[int, torch.Tensor],
    coeff: float,
    num_patches: int,
    max_samples: int,
    current_stats: Dict[str, Dict[str, Sequence[float]]],
    target_stats: Dict[str, Dict[str, Sequence[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    steering_hooks = MultiLayerSteeringHooks(llm_layers, layer_indices, vector_tokens, coeff, num_patches)
    baseline_losses: List[float] = []
    steered_losses: List[float] = []
    seen = 0
    total_batches = max(1, (max_samples + dataloader.batch_size - 1) // dataloader.batch_size)

    try:
        with torch.inference_mode():
            for raw_batch in tqdm(dataloader, total=total_batches, desc="eval-baseline+steered"):
                raw_batch = dict(raw_batch)
                raw_batch["_current_action_stats"] = current_stats["action"]
                batch = adapt_batch_normalization(raw_batch, current_stats, target_stats)
                prompt_input_ids, _ = build_prompt_only_inputs(batch)
                num_prompt_tokens = infer_num_prompt_tokens(prompt_input_ids)

                steering_hooks.set_num_prompt_tokens(None)
                baseline_losses.append(predict_loss(raw_batch, batch, cfg, vla, action_head, proprio_projector))

                steering_hooks.set_num_prompt_tokens(num_prompt_tokens)
                steered_losses.append(predict_loss(raw_batch, batch, cfg, vla, action_head, proprio_projector))
                steering_hooks.set_num_prompt_tokens(None)

                seen += raw_batch["input_ids"].shape[0]
                if seen >= max_samples:
                    break
    finally:
        steering_hooks.remove()

    return np.asarray(baseline_losses[:max_samples]), np.asarray(steered_losses[:max_samples])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    num_patches = compute_num_patches(vla, cfg)
    llm_layers, hook_layer_names = resolve_llm_layers(vla)
    layer_indices = infer_layer_indices(len(llm_layers), args.layers)

    print("Steering layers:")
    for idx in layer_indices:
        print(f"  [{idx}] {hook_layer_names[idx]}")

    vector_dataloader = build_dataloader(vla, processor, args, args.vector_dataset_name)
    vector_stats = vector_dataloader.dataset.dataset_statistics[args.vector_dataset_name]
    vector_tokens, raw_norms, quartile, num_vector_samples = compute_vectors(
        dataloader=vector_dataloader,
        cfg=cfg,
        vla=vla,
        action_head=action_head,
        proprio_projector=proprio_projector,
        llm_layers=llm_layers,
        layer_indices=layer_indices,
        num_patches=num_patches,
        max_samples=args.max_samples,
        current_stats=vector_stats,
        target_stats=vector_stats,
        normalize_vector=args.normalize_vector,
    )

    eval_dataloader = build_dataloader(vla, processor, args, args.eval_dataset_name)
    eval_stats = eval_dataloader.dataset.dataset_statistics[args.eval_dataset_name]
    baseline_losses, steered_losses = evaluate_baseline_and_steered(
        dataloader=eval_dataloader,
        cfg=cfg,
        vla=vla,
        action_head=action_head,
        proprio_projector=proprio_projector,
        llm_layers=llm_layers,
        layer_indices=layer_indices,
        vector_tokens=vector_tokens,
        coeff=args.coeff,
        num_patches=num_patches,
        max_samples=args.max_samples,
        current_stats=eval_stats,
        target_stats=vector_stats,
    )

    deltas = steered_losses - baseline_losses
    per_sample = pd.DataFrame(
        {
            "sample_index": np.arange(len(baseline_losses)),
            "baseline_loss": baseline_losses,
            "steered_loss": steered_losses,
            "loss_delta": deltas,
        }
    )
    per_sample.to_csv(output_dir / "per_sample.csv", index=False)

    summary = {
        "model_name": args.pretrained_checkpoint,
        "vector_dataset_name": args.vector_dataset_name,
        "eval_dataset_name": args.eval_dataset_name,
        "data_root_dir": args.data_root_dir,
        "selected_layers": layer_indices,
        "selected_layer_names": [hook_layer_names[idx] for idx in layer_indices],
        "coeff": args.coeff,
        "normalize_vector": args.normalize_vector,
        "vector_raw_norms": {str(idx): raw_norms[idx] for idx in layer_indices},
        "num_vector_samples": num_vector_samples,
        "num_eval_samples": int(len(baseline_losses)),
        "good_quartile_size": quartile,
        "bad_quartile_size": quartile,
        "baseline_average_loss": float(np.mean(baseline_losses)),
        "steered_average_loss": float(np.mean(steered_losses)),
        "average_loss_delta": float(np.mean(deltas)),
        "relative_change": float(np.mean(deltas) / np.mean(baseline_losses)),
        "improved_fraction": float(np.mean(deltas < 0)),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
