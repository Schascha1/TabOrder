# Core model release

Publication destination: `Schascha1/TabOrder`.

This directory contains the core models, synthetic training prior, compact
trainer, CPU example, tests, and two checkpoints. Full benchmark suites,
third-party baselines, real datasets, fine-tuning scripts, and experiment outputs
remain outside this release. This release supports model use; it is not a
complete experiment-reproduction package.

## Weights and provenance

| Public file | Research source |
| --- | --- |
| `taborder/model.py` | Model classes and topological divergence from `src/causal_foundation_model.py` |
| `taborder/dag_model.py` | Model classes and `masked_mean_std` from `src/causaldiscovery/causal_order_dag_model.py` |
| `taborder/data.py` | `VariableSpec` and `SyntheticDataset` from `src/datasets.py` |
| `taborder/train.py` | Compact adaptation of `src/train_model.py` |
| `checkpoints/taborder-gp-small.pth` | Tensor-identical export of `src/checkpoints/additive-unregularized-gp-small/final.pth` |
| `checkpoints/taborder-cc18-ctr23.pth` | Tensor-identical export of `src/checkpoints/finetuned_cc18_ctr23/final2.pth` |

The checkpoint manifest records source and export SHA-256 hashes, architectures,
evaluation settings, and missing training metadata. Optimizer state is omitted.
The fine-tuning notebook selects `final2.pth` for CC18/CTR23 evaluation; the other
file, `final.pth`, is not bundled. The notebook starts from the older mixed-prior
`METHOD_cfm_dfm-0_ep-25000_ft-0_gf-default_gn-default_gs-single_lo-0_vrng-5-10`
checkpoint. The released fine-tuned weights must not be described as a
fine-tuned version of the bundled additive GP model.

The two model variants deliberately preserve their distinct forward behavior.
The additive GP source includes later strict evaluation attention and cloned
parent-mask fixes. The fine-tuned DAG variant uses finite evaluation attention
bias, masked column-score normalization, and its original parent-attention head.
Both load strictly; no learned parameters are skipped or randomly filled.
Plotting, OpenML, and regression-pruning imports are removed from the core
modules. The additive model's input/output docstring is corrected.

## Which results are covered?

This table is based on checkpoint references in the research scripts and
notebooks. It establishes provenance, not a fresh benchmark replication.

| Result family | Coverage with these two weights |
| --- | --- |
| Jointly fine-tuned CC18/CTR23 results | `finetune_cc18_ctr23.ipynb` explicitly selects the bundled `final2.pth`. Reproduction still requires its data preparation, splits, masks, and downstream evaluation. |
| Additive GP synthetic order baseline | Bundled GP model is an active default in `order_discovery.py`. Non-additive GP and spline comparison rows need their own checkpoints. |
| Original Sachs intervention notebook | `sachs_ood.ipynb` selects the bundled additive GP model, then performs additional fine-tuning. The final adapted model is not bundled. |
| Original three-variable intervention notebook | Selects `non-additive-unregularized-gp-small`, which is not bundled. |
| Newer eight-variable intervention experiments | `rebuttal/03_interventions.ipynb` selects `additive-unregularized-gp-small-maskfix` and separate native ablation checkpoints. Not covered exactly. |
| Newer REGED environment results | `rebuttal/11_reged_environments.ipynb` selects `additive-unregularized-gp-small-maskfix`. Not covered exactly. |
| Robustness, diagnostics, and architecture ablations | Rebuttal notebooks select mask-fixed additive/non-additive models and multiple independently trained seeds. Not covered by these two weights. |
| Scalability comparisons | Model-size sweeps use additional checkpoints. Not covered by the small GP model alone. |
| Other-method baselines | Require the separate baseline implementations and evaluation setup. |

The mask-fixed pretrained files referenced above are absent from this checkout.
They are distinct training runs, not another name for the bundled GP weights.
There is no measured basis to claim that replacing them with these weights
reproduces the reported numbers or is quantitatively close enough.

## Validation and publication status

The original release preparation validated the additive model, generator,
training path, and portable installation with Python 3.10.12, PyTorch 2.7.1,
NumPy 2.2.6, and NetworkX 3.4.2. The expanded tests additionally verify the
fine-tuned model's strict loading, finite predictions, parameter count, and
both checkpoint checksums. The checkpoint tensors and model outputs are also
compared against the corresponding research implementations locally.

These checks are not full CC18/CTR23 or paper benchmark runs. GPU execution and
the full training recipe have not been validated for this preparation. PyTorch
2.7 emits deprecation warnings for mixed boolean/additive attention masks.

No code license or weight license has been selected; none has been invented.
Add the final citation when available. Publish only this standalone directory
to the public repository, keeping the unrelated research files and history out.
The weights total about 23 MB and can be stored directly in Git.
