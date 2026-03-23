from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from experiments.analysis.libero_sira_correlation import (
    build_cfg,
    build_dataloader,
    collect_losses_and_hiddens,
    compute_num_patches,
    compute_steering_vectors_from_hiddens,
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    infer_layer_indices,
    resolve_llm_layers,
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pairwise cosine similarity between OpenVLA steering vectors.")
    parser.add_argument("--data-root-dir", required=True)
    parser.add_argument(
        "--pretrained-checkpoint",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    )
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    parser.add_argument("--datasets", default="libero_goal_no_noops,libero_spatial_no_noops,libero_object_no_noops")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    parser.add_argument("--output-dir", default="outputs/libero_vector_similarity")
    parser.add_argument("--center-crop", action="store_true", default=True)
    return parser.parse_args()


def flatten_vector_tokens(vector: torch.Tensor) -> torch.Tensor:
    return vector.reshape(ACTION_DIM * NUM_ACTIONS_CHUNK, -1).reshape(-1).to(torch.float32)


def main() -> None:
    args = parse_args()
    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if len(dataset_names) < 2:
        raise ValueError("Need at least two datasets to compare steering vectors.")

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

    print("Vector-similarity hook layers:")
    for idx in layer_indices:
        print(f"  [{idx}] {hook_layer_names[idx]}")

    dataset_vectors: Dict[str, Dict[int, torch.Tensor]] = {}
    quartile_sizes: Dict[str, int] = {}

    for dataset_name in dataset_names:
        dataloader = build_dataloader(vla, processor, args, dataset_name)
        stats = dataloader.dataset.dataset_statistics[dataset_name]
        hiddens, losses = collect_losses_and_hiddens(
            dataloader=dataloader,
            cfg=cfg,
            vla=vla,
            action_head=action_head,
            proprio_projector=proprio_projector,
            num_patches=num_patches,
            layer_indices=layer_indices,
            max_samples=args.max_samples,
            desc=f"collect-vector-{dataset_name}",
            current_stats=stats,
            target_stats=stats,
        )
        vectors, quartile = compute_steering_vectors_from_hiddens(
            all_hiddens=hiddens,
            losses=losses,
            layer_indices=layer_indices,
        )
        dataset_vectors[dataset_name] = {idx: flatten_vector_tokens(vec) for idx, vec in vectors.items()}
        quartile_sizes[dataset_name] = quartile

    rows: List[dict] = []
    for layer_idx in layer_indices:
        for dataset_a, dataset_b in combinations(dataset_names, 2):
            vec_a = dataset_vectors[dataset_a][layer_idx]
            vec_b = dataset_vectors[dataset_b][layer_idx]
            cosine = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
            rows.append(
                {
                    "layer_index": layer_idx,
                    "layer_name": hook_layer_names[layer_idx],
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "cosine_similarity": float(cosine),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "pairwise_cosine.csv", index=False)

    summary = {
        "model_name": args.pretrained_checkpoint,
        "data_root_dir": args.data_root_dir,
        "datasets": dataset_names,
        "max_samples": args.max_samples,
        "selected_layers": layer_indices,
        "hook_layer_names": hook_layer_names,
        "quartile_sizes": quartile_sizes,
        "pairwise_cosine": rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
