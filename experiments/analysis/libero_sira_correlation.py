from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    NormalizationType,
)
from prismatic.vla.datasets.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.models.backbones.llm.prompting import PurePromptBuilder


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class SIRAHook:
    """ViNT-style hook that captures layer outputs directly."""

    def __init__(self) -> None:
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles = []

    def install(self, layers: Sequence[nn.Module]) -> None:
        for i, layer in enumerate(layers):
            def make_hook(idx: int):
                def hook_fn(module, inputs, output):
                    captured = output[0] if isinstance(output, tuple) else output
                    self.captured[idx] = captured
                return hook_fn

            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.captured.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIRA-style correlation analysis for OpenVLA-OFT on LIBERO RLDS.")
    parser.add_argument("--data-root-dir", required=True)
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--vector-dataset-name", default=None)
    parser.add_argument("--eval-dataset-name", default=None)
    parser.add_argument(
        "--pretrained-checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    )
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    parser.add_argument("--output-dir", default="outputs/libero_spatial_sira")
    parser.add_argument("--center-crop", action="store_true", default=True)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> SimpleNamespace:
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


def resolve_llm_layers(vla: nn.Module) -> tuple[List[nn.Module], List[str]]:
    candidates = []
    for name, module in vla.named_modules():
        match = re.search(r"language_model(?:\.[^.]+)*\.layers\.(\d+)$", name)
        if match:
            candidates.append((int(match.group(1)), name, module))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise RuntimeError("Could not resolve language model layers for SIRA hooking.")

    return [module for _, _, module in candidates], [name for _, name, _ in candidates]


def infer_layer_indices(num_layers: int, requested: str) -> List[int]:
    if requested == "all":
        return list(range(num_layers))
    indices = [int(item.strip()) for item in requested.split(",") if item.strip()]
    for idx in indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers")
    return indices


def flatten_hidden(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.reshape(hidden.shape[0], -1).float()


def build_dataloader(vla, processor, args: argparse.Namespace, dataset_name: str) -> DataLoader:
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    dataset = RLDSDataset(
        Path(args.data_root_dir),
        dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=1,
        train=True,
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    return DataLoader(dataset, batch_size=args.batch_size, sampler=None, collate_fn=collator, num_workers=0)


def compute_num_patches(vla, cfg) -> int:
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        num_patches += 1
    if cfg.use_diffusion:
        num_patches += 1
    return num_patches


def unnormalize_bounds_tensor(
    values: torch.Tensor,
    stats: Dict[str, Sequence[float]],
) -> torch.Tensor:
    low = torch.as_tensor(stats["q01"], dtype=values.dtype, device=values.device)
    high = torch.as_tensor(stats["q99"], dtype=values.dtype, device=values.device)
    return 0.5 * (values + 1.0) * (high - low + 1e-8) + low


def renormalize_bounds_tensor(
    values: torch.Tensor,
    from_stats: Dict[str, Sequence[float]],
    to_stats: Dict[str, Sequence[float]],
) -> torch.Tensor:
    raw = unnormalize_bounds_tensor(values, from_stats)
    to_low = torch.as_tensor(to_stats["q01"], dtype=values.dtype, device=values.device)
    to_high = torch.as_tensor(to_stats["q99"], dtype=values.dtype, device=values.device)
    renorm = 2.0 * (raw - to_low) / (to_high - to_low + 1e-8) - 1.0
    renorm = torch.clamp(renorm, -1.0, 1.0)

    zeros_mask = to_low == to_high
    if zeros_mask.any():
        renorm = torch.where(zeros_mask, torch.zeros_like(renorm), renorm)
    return renorm


def adapt_batch_normalization(
    batch: Dict[str, torch.Tensor],
    current_stats: Dict[str, Dict[str, Sequence[float]]],
    target_stats: Dict[str, Dict[str, Sequence[float]]],
) -> Dict[str, torch.Tensor]:
    if ACTION_PROPRIO_NORMALIZATION_TYPE != NormalizationType.BOUNDS_Q99:
        raise NotImplementedError(
            f"Cross-dataset renormalization currently expects bounds_q99, got {ACTION_PROPRIO_NORMALIZATION_TYPE}."
        )

    adapted = dict(batch)
    adapted["actions"] = renormalize_bounds_tensor(batch["actions"], current_stats["action"], target_stats["action"])
    if batch.get("proprio") is not None:
        adapted["proprio"] = renormalize_bounds_tensor(batch["proprio"], current_stats["proprio"], target_stats["proprio"])
    return adapted


def build_prompt_only_inputs(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("Predictive SIRA analysis currently requires --batch-size 1.")
    labels = batch["labels"][0]
    action_positions = torch.nonzero(labels > ACTION_TOKEN_BEGIN_IDX, as_tuple=False)
    if action_positions.numel() == 0:
        raise RuntimeError("Could not locate action-token suffix in labels.")
    first_action_idx = int(action_positions[0].item())
    prompt_input_ids = batch["input_ids"][:, :first_action_idx].clone()
    prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=batch["attention_mask"].dtype)
    return prompt_input_ids, prompt_attention_mask


def infer_num_prompt_tokens(prompt_input_ids: torch.Tensor) -> int:
    trailing_action_start_token = 29871
    if int(prompt_input_ids[0, -1].item()) == trailing_action_start_token:
        return prompt_input_ids.shape[1] - 1
    return prompt_input_ids.shape[1]


def extract_predictive_hidden_from_layer(
    hidden: torch.Tensor,
    num_patches: int,
    num_prompt_tokens: int,
) -> torch.Tensor:
    actions_hidden_states = hidden[
        :,
        num_patches + num_prompt_tokens : num_patches + num_prompt_tokens + ACTION_DIM * NUM_ACTIONS_CHUNK,
        :,
    ]
    return actions_hidden_states.to(torch.bfloat16)


def collect_losses_and_hiddens(
    dataloader: DataLoader,
    cfg,
    vla,
    action_head,
    proprio_projector,
    num_patches: int,
    layer_indices: List[int],
    max_samples: int,
    desc: str,
    current_stats: Dict[str, Dict[str, Sequence[float]]],
    target_stats: Dict[str, Dict[str, Sequence[float]]],
) -> tuple[Dict[int, torch.Tensor], np.ndarray]:
    llm_layers, _ = resolve_llm_layers(vla)
    hook = SIRAHook()
    hook.install(llm_layers)

    all_hiddens = {i: [] for i in layer_indices}
    all_losses = []
    seen = 0
    total_batches = max(1, (max_samples + dataloader.batch_size - 1) // dataloader.batch_size)

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=total_batches, desc=desc):
            batch = adapt_batch_normalization(batch, current_stats, target_stats)
            prompt_input_ids, prompt_attention_mask = build_prompt_only_inputs(batch)
            num_prompt_tokens = infer_num_prompt_tokens(prompt_input_ids)

            proprio = None
            if cfg.use_proprio and batch.get("proprio") is not None:
                proprio = batch["proprio"].to(torch.bfloat16)

            predicted_actions_raw, _ = vla.predict_action(
                input_ids=prompt_input_ids.to(DEVICE),
                attention_mask=prompt_attention_mask.to(DEVICE),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(DEVICE),
                proprio=proprio,
                proprio_projector=proprio_projector,
                action_head=action_head,
                unnorm_key=cfg.unnorm_key,
                use_film=False,
            )
            predicted_actions = torch.tensor(predicted_actions_raw, dtype=torch.float32).unsqueeze(0)
            ground_truth_actions = unnormalize_bounds_tensor(batch["actions"], current_stats["action"]).to(torch.float32)
            batch_losses = torch.abs(predicted_actions - ground_truth_actions).mean(dim=(1, 2)).cpu()
            all_losses.append(batch_losses)

            for layer_idx in layer_indices:
                actions_hidden = extract_predictive_hidden_from_layer(
                    hook.captured[layer_idx], num_patches, num_prompt_tokens
                )
                all_hiddens[layer_idx].append(flatten_hidden(actions_hidden.detach().cpu()).to(torch.float16))

            seen += batch["input_ids"].shape[0]
            if seen >= max_samples:
                break

    hook.remove()

    losses = torch.cat(all_losses, dim=0)[:max_samples]
    for layer_idx in layer_indices:
        all_hiddens[layer_idx] = torch.cat(all_hiddens[layer_idx], dim=0)[:max_samples]

    return all_hiddens, losses.numpy()


def compute_steering_vectors_from_hiddens(
    all_hiddens: Dict[int, torch.Tensor],
    losses: np.ndarray,
    layer_indices: List[int],
) -> tuple[Dict[int, torch.Tensor], int]:
    losses_t = torch.from_numpy(losses)
    sorted_indices = torch.argsort(losses_t)
    q = max(1, len(sorted_indices) // 4)
    good_indices = sorted_indices[:q]
    bad_indices = sorted_indices[len(sorted_indices) - q :]

    vectors = {}
    for layer_idx in layer_indices:
        h = all_hiddens[layer_idx].to(torch.float32)
        good_mean = h[good_indices].mean(dim=0)
        bad_mean = h[bad_indices].mean(dim=0)
        v = good_mean - bad_mean
        v_normalized = v / (v.norm() + 1e-12)
        vectors[layer_idx] = v_normalized
        cos_sim = torch.nn.functional.cosine_similarity(good_mean.unsqueeze(0), bad_mean.unsqueeze(0)).item()
        print(f"  SIRA L{layer_idx}: raw_norm={v.norm().item():.2f}, cos_sim={cos_sim:.4f}")

    print(f"  SIRA: {q} good, {q} bad samples (middle {len(sorted_indices) - 2*q} ignored) from {len(losses_t)} valid")
    return vectors, q


def compute_cosine_alignments_from_hiddens(
    all_hiddens: Dict[int, torch.Tensor],
    layer_indices: List[int],
    steering_vectors: Dict[int, torch.Tensor],
) -> Dict[int, np.ndarray]:
    alignments = {}
    for layer_idx in layer_indices:
        hidden = torch.nn.functional.normalize(all_hiddens[layer_idx].to(torch.float32), dim=1)
        steering = steering_vectors[layer_idx].unsqueeze(0)
        cosine_values = torch.matmul(hidden, steering.T).squeeze(1).numpy()
        alignments[layer_idx] = cosine_values.astype(np.float64, copy=False)
    return alignments


def summarize_correlations(losses: np.ndarray, alignments: Dict[int, np.ndarray]) -> List[dict]:
    rows = []
    for layer_idx, cos_values in alignments.items():
        valid = np.isfinite(losses) & np.isfinite(cos_values)
        spearman = spearmanr(losses[valid], cos_values[valid])
        kendall = kendalltau(losses[valid], cos_values[valid])
        rows.append(
            {
                "layer_index": layer_idx,
                "num_samples": int(valid.sum()),
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_pvalue": float(kendall.pvalue),
            }
        )
    rows.sort(key=lambda row: row["spearman_rho"])
    return rows


def main() -> None:
    args = parse_args()
    vector_dataset_name = args.vector_dataset_name or args.dataset_name
    eval_dataset_name = args.eval_dataset_name or args.dataset_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    vla = get_vla(cfg)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    num_patches = compute_num_patches(vla, cfg)
    hook_layers, hook_layer_names = resolve_llm_layers(vla)
    layer_indices = infer_layer_indices(len(hook_layers), args.layers)

    print("SIRA hook layers:")
    for idx in layer_indices:
        print(f"  [{idx}] {hook_layer_names[idx]}")

    vector_dataloader = build_dataloader(vla, processor, args, vector_dataset_name)
    vector_stats = vector_dataloader.dataset.dataset_statistics[vector_dataset_name]
    vector_hiddens, vector_losses = collect_losses_and_hiddens(
        dataloader=vector_dataloader,
        cfg=cfg,
        vla=vla,
        action_head=action_head,
        proprio_projector=proprio_projector,
        num_patches=num_patches,
        layer_indices=layer_indices,
        max_samples=args.max_samples,
        desc=f"collect-vector-{vector_dataset_name}",
        current_stats=vector_stats,
        target_stats=vector_stats,
    )

    steering_vectors, quartile = compute_steering_vectors_from_hiddens(
        all_hiddens=vector_hiddens,
        losses=vector_losses,
        layer_indices=layer_indices,
    )

    eval_dataloader = build_dataloader(vla, processor, args, eval_dataset_name)
    eval_stats = eval_dataloader.dataset.dataset_statistics[eval_dataset_name]
    eval_hiddens, eval_losses = collect_losses_and_hiddens(
        dataloader=eval_dataloader,
        cfg=cfg,
        vla=vla,
        action_head=action_head,
        proprio_projector=proprio_projector,
        num_patches=num_patches,
        layer_indices=layer_indices,
        max_samples=args.max_samples,
        desc=f"collect-eval-{eval_dataset_name}",
        current_stats=eval_stats,
        target_stats=vector_stats,
    )
    alignments = compute_cosine_alignments_from_hiddens(eval_hiddens, layer_indices, steering_vectors)

    summary_rows = summarize_correlations(eval_losses, alignments)
    per_sample = pd.DataFrame({"sample_index": np.arange(len(eval_losses)), "loss": eval_losses})
    for layer_idx in layer_indices:
        per_sample[f"cosine_layer_{layer_idx}"] = alignments[layer_idx]
    per_sample.to_csv(output_dir / "per_sample.csv", index=False)

    summary = {
        "model_name": args.pretrained_checkpoint,
        "dataset_name": args.dataset_name,
        "vector_dataset_name": vector_dataset_name,
        "eval_dataset_name": eval_dataset_name,
        "data_root_dir": args.data_root_dir,
        "max_samples": int(len(eval_losses)),
        "hook_layer_names": hook_layer_names,
        "selected_layers": layer_indices,
        "good_quartile_size": quartile,
        "bad_quartile_size": quartile,
        "correlations": summary_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
