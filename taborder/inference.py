"""Encode continuous tables and impute without using hidden ground truth."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EncodedTable:
    x: torch.Tensor
    row_mask: torch.Tensor
    col_mask: torch.Tensor
    query: torch.Tensor
    mean: np.ndarray
    scale: np.ndarray


def encode_table(values, missing, *, context_size: int, device="cpu") -> EncodedTable:
    """Fit scaling on observed context entries; fill missing entries with that mean.

    `missing=True` means unavailable. Hidden entries may contain NaN or arbitrary
    ground truth: neither changes the encoded input. Every column must have at
    least one observed context entry. Input and output columns stay in order.
    """
    values = np.asarray(values, dtype=np.float64)
    missing = np.asarray(missing, dtype=bool)
    if values.ndim != 2 or values.shape != missing.shape or values.shape[1] < 2:
        raise ValueError("values and missing must have matching (rows, columns) shapes with >=2 columns")
    n, c = values.shape
    if not 1 <= context_size <= n:
        raise ValueError("context_size must be between 1 and the number of rows")
    if not np.isfinite(values[~missing]).all():
        raise ValueError("Observed values must be finite; flag unavailable entries in missing")
    mean, scale = np.empty(c), np.empty(c)
    for j in range(c):
        observed = values[:context_size, j][~missing[:context_size, j]]
        if not len(observed):
            raise ValueError(f"Column {j} has no observed context values")
        mean[j] = observed.mean()
        std = observed.std(ddof=1) if len(observed) > 1 else 0.0
        scale[j] = std if std > 1e-6 else 1.0
    normalized = (np.where(missing, mean, values) - mean) / scale
    x = torch.zeros((1, n, c, 4), dtype=torch.float32, device=device)
    x[0, ..., 0] = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    x[0, ..., 1] = torch.as_tensor(missing, dtype=torch.float32, device=device)
    row_mask = torch.zeros((1, n), dtype=torch.bool, device=device)
    col_mask = torch.zeros((1, c), dtype=torch.bool, device=device)
    query = torch.arange(n, device=device)[None, :] >= context_size
    return EncodedTable(x, row_mask, col_mask, query, mean, scale)


def impute_table(model, values, missing, *, context_size: int, order_scores_override=None):
    """Return imputed values, predicted means, and inferred order scores.

    All values are returned in original units. Observed values are preserved.
    An optional order override controls the mask; it is never inferred from a
    ground-truth graph here. Both released model variants accept this argument.
    """
    device = next(model.parameters()).device
    table = encode_table(values, missing, context_size=context_size, device=device)
    if order_scores_override is not None:
        order_scores_override = torch.as_tensor(order_scores_override, dtype=table.x.dtype, device=device)
        if order_scores_override.shape != (1, table.x.shape[2]) or not order_scores_override.isfinite().all():
            raise ValueError("order_scores_override must be finite with shape (1, columns)")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            outputs = model(table.x, row_mask=table.row_mask, col_mask=table.col_mask,
                            query_indicator=table.query, order_scores_override=order_scores_override)
    finally:
        model.train(was_training)
    predicted = outputs[4][0, ..., 0].cpu().numpy() * table.scale + table.mean
    if not np.isfinite(predicted).all():
        raise RuntimeError("Model returned non-finite predictions")
    return {
        "imputed": np.where(missing, predicted, values),
        "predicted_mean": predicted,
        "order_scores": outputs[2].detach().cpu().numpy()[0],
    }
