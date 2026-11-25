"""Strict loading for the released TabOrder architecture."""

from pathlib import Path

import torch

from .model import CausalOrderModel
from .dag_model import CausalOrderDAGModel


def load_model(path: str | Path, device: str | torch.device = "cpu") -> CausalOrderModel | CausalOrderDAGModel:
    """Load a release checkpoint or a matching legacy training checkpoint.

    The release metadata records non-parameter settings such as temperature.
    Legacy state dictionaries use the evaluation defaults from the benchmarks.
    Incompatible architectures raise an error instead of silently skipping keys.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary.")
    state_dict = checkpoint
    for key in ("model_state_dict", "model", "state_dict"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    kind = checkpoint.get("model_kind", "order" if any(key.startswith("edge_mlp.") for key in state_dict) else "dag")
    if kind not in {"order", "dag"}:
        raise ValueError(f"Unsupported model kind: {kind!r}")
    defaults = {
        "d_model": 4,
        "d_classifier": 10,
        "d_embedding": 128,
        "K_attention_blocks_order": 6,
        "K_attention_blocks_pred": 4,
        "temperature_order": 0.1,
        "attention_bias": -20.0,
        "order_mode": "causal",
    }
    if kind == "order":
        defaults["fix_inplace"] = True
    config = checkpoint.get("model_config", defaults)
    model = (CausalOrderModel if kind == "order" else CausalOrderDAGModel)(**config)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
