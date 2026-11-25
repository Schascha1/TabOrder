"""Generate a GP table and infer its order and missing values on CPU."""

import argparse
from pathlib import Path

import torch

from taborder import SyntheticDataset, load_model, topological_divergence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/taborder-gp-small.pth"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    dataset = SyntheticDataset(
        variable_mode="additive-gp", variable_range=(5, 10), parent_range=(1, 3),
        noise_var_range=(0.1, 0.5), batch_size=1,
        min_seq_len=128, max_seq_len=128, seed=42,
    )
    batch = next(iter(dataset))
    x, row_mask, col_mask, truth, query, dags, masked, *_ = batch
    model = load_model(args.checkpoint, args.device)
    with torch.inference_mode():
        outputs = model(x.to(args.device), row_mask=row_mask.to(args.device),
                        col_mask=col_mask.to(args.device), query_indicator=query.to(args.device))
    order_scores = outputs[2].cpu()
    prediction = outputs[4].cpu()
    mse = (prediction[..., 0][masked] - truth[masked]).square().mean()
    print("Inferred order (earliest first):", order_scores[0].argsort().tolist())
    print("Topological divergence:", topological_divergence(order_scores[0], dags[0]))
    print("Masked-cell MSE (normalized units):", mse.item())
    print("Prediction shape (batch, rows, columns, mean/log_variance):", tuple(prediction.shape))


if __name__ == "__main__":
    main()
