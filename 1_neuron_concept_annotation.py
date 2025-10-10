import os
import json
import pickle
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F


def load_any_pkl(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def normalize_torch(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def compute_representative_concepts(
    representative_sequences: List[torch.Tensor],
    concept_names: List[str],
    concept_emb_np: np.ndarray,
    save_json_path: str = "representative_concepts_by_neuron.json",
) -> List[Dict[str, Any]]:
    """
    Args:
        representative_sequences: list of tensors per neuron; each tensor has shape (num_features, 768)
        concept_names: list of C (=270) concept strings
        concept_emb_np: (C, 768) float32 numpy array of concept embeddings

    Returns:
        A list of dicts, one per neuron, including selected concepts and scores.
    """
    C = len(concept_names)
    assert concept_emb_np.ndim == 2 and concept_emb_np.shape[0] == C, "concept_emb shape mismatch"
    D = concept_emb_np.shape[1]

    # L2-normalize concept embeddings
    concept_emb = torch.from_numpy(concept_emb_np.astype(np.float32))
    concept_emb = normalize_torch(concept_emb, dim=1)  # (C, D)

    results: List[Dict[str, Any]] = []

    for n_idx, feats in enumerate(representative_sequences):
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        feats = feats.float().contiguous()  # (F, D) or (0, D)

        if feats.numel() == 0:
            results.append({
                "neuron_idx": n_idx,
                "num_features": 0,
                "threshold": None,
                "min": None,
                "max": None,
                "selected_count": 0,
                "selected_concepts_sorted": [],  # name & score
                "selected_indices_sorted": [],
                "selected_scores_sorted": [],
            })
            print(f"[neuron {n_idx:03d}] empty features -> selected 0 concepts")
            continue

        assert feats.shape[1] == D, f"feature dim mismatch at neuron {n_idx}: {feats.shape} vs D={D}"

        # Feature normalization
        feats = normalize_torch(feats, dim=1)  # (F, D)

        # Cosine similarity: (F x C) = (F x D) @ (D x C)
        sims = feats @ concept_emb.t()  # (F, C)

        # Per-concept mean similarity
        avg_sim = sims.mean(dim=0)  # (C,)

        vmin = float(avg_sim.min().item())
        vmax = float(avg_sim.max().item())
        # Threshold at max - 0.05 * (max - min)
        thr = vmax - 0.05 * (vmax - vmin)

        sel_mask = avg_sim >= thr
        sel_idx = sel_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        # Sort selected concepts by mean cosine similarity (descending)
        sel_idx_sorted = sorted(sel_idx, key=lambda i: float(avg_sim[i].item()), reverse=True)
        sel_scores_sorted = [float(avg_sim[i].item()) for i in sel_idx_sorted]
        sel_concepts_sorted = [
            {"index": i, "name": concept_names[i], "avg_cos_sim": float(avg_sim[i].item())}
            for i in sel_idx_sorted
        ]

        # (Optional) If you want to store all concepts sorted by score, uncomment below.
        # all_idx_sorted = torch.argsort(avg_sim, descending=True).tolist()
        # all_concepts_sorted = [
        #     {"index": i, "name": concept_names[i], "avg_cos_sim": float(avg_sim[i].item())}
        #     for i in all_idx_sorted
        # ]

        results.append({
            "neuron_idx": n_idx,
            "num_features": int(feats.shape[0]),
            "threshold": thr,
            "min": vmin,
            "max": vmax,
            "selected_count": len(sel_idx_sorted),
            "selected_concepts_sorted": sel_concepts_sorted,  # includes names and scores
            "selected_indices_sorted": sel_idx_sorted,        # sorted indices
            "selected_scores_sorted": sel_scores_sorted,      # sorted scores
            # "all_concepts_sorted": all_concepts_sorted,     # enable if needed
        })

        top1_name = sel_concepts_sorted[0]["name"] if sel_concepts_sorted else "—"
        top1_score = sel_concepts_sorted[0]["avg_cos_sim"] if sel_concepts_sorted else float("nan")
        print(
            f"[neuron {n_idx:03d}] F={feats.shape[0]} | min={vmin:.4f} max={vmax:.4f} thr={thr:.4f} "
            f"| selected={len(sel_idx_sorted)} | top1={top1_name}({top1_score:.4f})"
        )

    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    # Save results
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] Saved representative concept sets -> {save_json_path}")

    return results


if __name__ == "__main__":
    spr_model = "ASFormer"
    selection_method = "4_Video_wise_Top1"
    concept_sets = "ChoLec-270"

    concept_pkl = f"extracted_features/{spr_model}/text/{concept_sets}.pkl"
    concept_payload = load_any_pkl(concept_pkl)
    concept_names: List[str] = concept_payload["concepts"]              # length 270
    concept_emb_np: np.ndarray = concept_payload["embeddings"]          # (270, 768)

    # Load list of representative sequence features per neuron
    seq_pkl = f"extracted_features/{spr_model}/sequence/{selection_method}.pkl"
    representative_sequences = load_any_pkl(seq_pkl)  # list of tensors (F_i, 768)

    # Compute and save
    _ = compute_representative_concepts(
        representative_sequences=representative_sequences,
        concept_names=concept_names,
        concept_emb_np=concept_emb_np,
        save_json_path=f"extracted_neuron_concepts/{spr_model}/{selection_method}_and_{concept_sets}.json",
    )
