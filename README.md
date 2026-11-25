# TabOrder

Core research code for learning causal orders and predicting missing values in
tabular data. This initial release contains the TabOrder model, its synthetic
Gaussian-process SCM training prior, an additive GP checkpoint, and weights
fine-tuned jointly on CC18 and CTR23.

## Install and run

Python 3.10 or newer is required. Run these commands from this directory:

```sh
python -m pip install -e .
python examples/quickstart.py
python -m unittest discover -s tests -v
```

The example runs on CPU and generates its own data. The only runtime dependencies
are PyTorch, NumPy, and NetworkX. GPU execution is available with
`python examples/quickstart.py --device cuda` and a CUDA-enabled PyTorch install.

```python
import torch
from taborder import SyntheticDataset, load_model

torch.manual_seed(42)
data = SyntheticDataset(
    variable_mode="additive-gp", variable_range=(5, 10),
    parent_range=(1, 3), noise_var_range=(0.1, 0.5),
    batch_size=1, min_seq_len=128, max_seq_len=128, seed=42,
)
x, row_mask, col_mask, truth, query, dags, masked, *_ = next(iter(data))
model = load_model("checkpoints/taborder-gp-small.pth")
with torch.inference_mode():
    outputs = model(x, row_mask=row_mask, col_mask=col_mask, query_indicator=query)
scores = outputs[2]                 # (batch, columns); smaller means earlier
order = scores.argsort(dim=-1)
mean = outputs[4][..., 0]           # (batch, rows, columns)
variance = outputs[4][..., 1].exp()
```

## Model and data interfaces

`CausalOrderModel` takes a tensor of shape `(batch, rows, columns, 4)`.
The four channels hold the value, missing-value indicator, categorical flag,
and number of categories. The released synthetic prior generates continuous
variables, so the last two channels are zero. Missing values are filled with
context means and separately flagged. Row and column padding masks are boolean,
with `True` meaning padding; provide both masks, including for unpadded tables.

The seven forward outputs are edge scores, pairwise order probabilities, order
scores, missing-parent scores, mean/log-variance predictions, variance-bias
terms, and parent attention weights. Parent matrices use `[child, parent]`.
Attention weights are model internals, rather than calibrated edge probabilities.
`query_indicator` is accepted for compatibility; the current architecture does
not use it to enforce context/query attention separation.

`SyntheticDataset` yields the original 12-item training tuple: encoded input,
row padding, column padding, unmasked values, query indicator, DAGs, missing-cell
mask, ancestor indicator, valid-cell indicator, parent availability, target
indices, and column types. Ancestor indicators use `[ancestor, descendant]`.
The generator supports additive and non-additive GP mechanisms, approximated by
256 random Fourier features, as well as the existing spline mode.

The batch shares one sampled DAG, while each table samples its mechanisms.
`seed` controls NumPy draws; also set `torch.manual_seed` for PyTorch draws.
Context normalization fits each column on its observed context entries. Outputs
and ground truth are in normalized units. The legacy context-mean fill is
preserved, including its use of synthetic ground-truth context values; this
generator is a training prior, not a preprocessing API for real missing data.

## Checkpoints

| File | Purpose | Model variant |
| --- | --- | --- |
| `checkpoints/taborder-gp-small.pth` | Unregularized additive GP pretraining | `CausalOrderModel` |
| `checkpoints/taborder-cc18-ctr23.pth` | Joint CC18/CTR23 fine-tuning, exported from `final2.pth` | `CausalOrderDAGModel` |

Both use `load_model`, which selects the recorded architecture and loads every
parameter strictly. The fine-tuned model has 2,767,888 parameters. It descends
from an older mixed-prior pretrained model, not the additive GP checkpoint.
Its model variant retains the finite evaluation attention bias used by the
CC18/CTR23 notebook; the additive GP model uses the current strict evaluation
mask. These variants share an input/output interface, but their forward passes
are not interchangeable.

```python
model = load_model("checkpoints/taborder-cc18-ctr23.pth")
```

To check loading and inference with synthetic inputs:

```sh
python examples/quickstart.py --checkpoint checkpoints/taborder-cc18-ctr23.pth
```

This is an inference smoke check, not a CC18/CTR23 benchmark evaluation.

`checkpoints/taborder-gp-small.pth` exports the tensor weights from
`additive-unregularized-gp-small/final.pth`. It has 2,899,473 parameters, six order
blocks, four prediction blocks, 128-dimensional embeddings, and a recorded
training step of 24,999 (25,000 updates). Optimizer and scheduler state have been
removed. Evaluation settings are stored with the weights; loading is strict.
Hashes and source identifiers are recorded in `checkpoints/manifest.json`.

Training scripts associate this checkpoint with additive GP data, 5–10 variables,
and no order supervision (`lambda_order=0`). The legacy checkpoint does not
record a full training configuration or seed. These two weights support a core
model release; they do not reproduce the entire paper. In particular, newer
rebuttal experiments use separately trained mask-fixed checkpoints, and ablations
use additional models. No numerical closeness to those results has been
established. See `RELEASE_NOTES.md` for the result coverage and evidence.

## Train

```sh
python -m taborder.train --output runs/gp-small.pth
```

The default recipe uses additive GP data, 5–10 variables, 512–1024 rows,
20% MCAR masking, Gaussian NLL, AdamW, and 25,000 updates. To check the training
path on CPU:

```sh
python -m taborder.train --steps 2 --batch-size 1 \
  --min-seq-len 32 --max-seq-len 32 --device cpu --output runs/smoke.pth
python examples/quickstart.py --checkpoint runs/smoke.pth
```

This compact trainer starts a new run and saves inference weights and its
configuration. It preserves the core objective and schedule while using full
precision, single-process data generation, and the existing autograd fix.
It does not reproduce the old checkpoint bit for bit or support resuming.
