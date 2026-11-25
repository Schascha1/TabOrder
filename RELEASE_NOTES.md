# Core model release

Public repository: https://github.com/Schascha1/TabOrder

The release focuses on synthetic causal-order recovery, missing-value imputation,
and intervention prediction. Start with `examples/intro.ipynb`; its default run
is offline and uses the two bundled checkpoints. The full benchmark collection,
third-party baseline code, and unrelated ablation models are outside this scope.

## Notebook and Figure 1

| Experiment | What runs |
| --- | --- |
| Figure 1 | Reconstruct the original two panels from all 500 saved paper predictions. Same plotted values, with Matplotlib layout rather than TikZ. |
| Synthetic order | Fresh additive GP tables with five variables and 0%, 20%, and 40% missingness; compare inferred orders with random orders across five seeds. |
| Imputation | Fresh synthetic GP tables and the offline diabetes feature table; compare context mean, KNN, GP weights, and CC18/CTR23 weights on the same missing query cells across three seeds. |
| Intervention | Fresh paired observational/intervened queries in the original example's spline chain, using seed 20 and the bundled additive GP model. Infer and freeze an order from context; compare with an explicitly labelled random forest baseline. |
| Optional TabPFN | Opt-in fresh inference, requiring a separate compatible checkpoint or model download. It is not run by default. |

The Figure 1 data come from
`paper/expres/three_variable_intervention_predictions.csv`; a checksum and
source settings are in `examples/data/figure1_manifest.json`. The corresponding
research notebook is `src/experiments/notebooks/three_variable_intervention_failure.ipynb`.
That notebook selects the non-additive GP checkpoint, which is not bundled.
The historical TabPFN version was not recorded in the CSV, so the optional
fresh comparison is not presented as an exact numerical rerun of Figure 1.

The live intervention uses paired query rows with identical X and Y, changing
only query Z. It also uses fewer rows and freezes an observational-context order,
so its metrics are distinct from Figure 1's saved mixed-regime query predictions.
The known chain order is not supplied to the model. The learned order is printed,
including if it is wrong; seeds and results are not selected to force success.

The imputation example is a compact model-use demonstration, not a reproduction
of CC18/CTR23 aggregate ranks. Scaling and KNN fitting use observed context only;
hidden truth is replaced with NaN before encoding. The offline diabetes table is
not asserted to have been excluded from historical fine-tuning. Reproducing the
full CC18/CTR23 tables would also require their data preparation and downstream
models, not only the weights.

## Weights and source provenance

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
The fine-tuning notebook selects `final2.pth` and starts from the older mixed-prior
`METHOD_cfm_dfm-0_ep-25000_ft-0_gf-default_gn-default_gs-single_lo-0_vrng-5-10`
checkpoint. The released fine-tuned weights therefore do not descend from the
bundled additive GP model.

The two model variants preserve their distinct default forward behavior.
The additive GP source includes the later strict evaluation attention and cloned
parent-mask fixes. The fine-tuned DAG variant retains finite attention bias and
masked column-score normalization. Both load strictly, without skipping learned
parameters. The public additive model additionally accepts an optional order
score override, matching the DAG variant's existing interface. This enables
context-frozen intervention prediction and changes no checkpoint tensors.

`taborder/inference.py` adds observed-context-only normalization and mean filling
for actual tables. This is separate from the legacy synthetic generator, whose
historical behavior is retained. Plotting, OpenML, and regression-pruning imports
have been removed from the core model modules. Notebook dependencies are optional.

## Validation

Core tests cover checkpoint hashes and loading, finite predictions, padding,
generator reproducibility, gradient flow, hidden-ground-truth independence,
observed-value preservation, and blocking descendant effects with a fixed mask.
The default notebook is executed end to end locally; CI also executes it and
uploads the generated figures, metrics, and executed notebook as an artifact.
The optional fresh TabPFN path, GPU execution, full CC18/CTR23 benchmark, and
full pretraining recipe have not been run for this notebook update.

No code license or weight license has been selected. The weights total about
23 MB and can be stored directly in Git.
