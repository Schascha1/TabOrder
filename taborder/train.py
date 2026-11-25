"""Train the core GP model; saves weights and configuration for inference."""

import argparse
from pathlib import Path

import torch

from .data import SyntheticDataset
from .model import CausalOrderModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/gp-small.pth"))
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-seq-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--variable-mode", choices=["additive-gp", "non-additive-gp"], default="additive-gp")
    parser.add_argument("--variable-range", type=int, nargs=2, default=(5, 10))
    parser.add_argument("--lambda-order", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 2 or args.batch_size < 1:
        parser.error("--steps must be >= 2 and --batch-size must be positive")
    if args.min_seq_len < 4 or args.max_seq_len < args.min_seq_len:
        parser.error("sequence lengths must satisfy 4 <= min <= max")
    if not 2 <= args.variable_range[0] <= args.variable_range[1]:
        parser.error("variable range must satisfy 2 <= min <= max")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataset = SyntheticDataset(
        variable_mode=args.variable_mode, variable_range=tuple(args.variable_range),
        parent_range=(1, 3), noise_var_range=(0.1, 0.5),
        batch_size=args.batch_size, min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len, seed=args.seed,
    )
    iterator = iter(dataset)
    config = dict(d_model=4, d_classifier=10, d_embedding=128,
                  K_attention_blocks_order=6, K_attention_blocks_pred=4,
                  temperature_order=1.0, attention_bias=-5.0,
                  fix_inplace=True, order_mode="causal")
    model = CausalOrderModel(**config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=1e-2)
    warmup = max(1, int(0.03 * args.steps))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps - warmup, eta_min=1e-5),
        ], milestones=[warmup],
    )
    for step in range(args.steps):
        fraction = step / (args.steps - 1)
        model.temperature_order = 1.0 - 0.9 * fraction
        model.attention_bias = -5.0 - 15.0 * fraction
        batch = next(iterator)
        x, row_mask, col_mask, truth, query, _, masked, ancestors, *_ = [
            item.to(device) if torch.is_tensor(item) else item for item in batch
        ]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, row_mask=row_mask, col_mask=col_mask, query_indicator=query)
        prediction = outputs[4]
        if not masked.any():
            raise RuntimeError("No masked cells in batch; increase sequence length or batch size.")
        loss = torch.nn.functional.gaussian_nll_loss(
            prediction[..., 0][masked], truth[masked], prediction[..., 1][masked].exp(),
        )
        if args.lambda_order and ancestors.any():
            loss = loss + args.lambda_order * outputs[1][ancestors].mean()
        if not loss.isfinite():
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={loss.item():.6f}")
    config.update(temperature_order=model.temperature_order, attention_bias=model.attention_bias)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save({"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "model_config": config, "training_config": training_config, "step": args.steps - 1}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
