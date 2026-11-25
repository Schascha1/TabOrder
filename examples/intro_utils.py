"""Small evaluation helpers for the introductory notebook.

The chain/spline mechanisms are extracted from the original Figure 1 notebook.
"""

import math
import numpy as np


def _gp_rff(parents, rng, hidden_dim=256):
    """Random Fourier Features for an RBF GP-like prior.
    parents: (n, P)
    returns: (n,)
    """
    n, P = parents.shape
    if P == 0:
        return np.zeros((n,), dtype=float)

    x = parents.astype(float, copy=True)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / (x.std(axis=0, keepdims=True) + 1e-6)

    lengthscale = float(np.exp(rng.uniform(np.log(0.3), np.log(1.0))))
    W = rng.normal(size=(P, hidden_dim)) / lengthscale
    b = rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,))
    phi = math.sqrt(2.0 / hidden_dim) * np.cos(x @ W + b)
    w = rng.normal(size=(hidden_dim,))
    return phi @ w


def _random_cubic_spline_1d(z: np.ndarray, rng: np.random.Generator, K: int = 6) -> np.ndarray:
    """
    Smooth random cubic Hermite spline on u = sigmoid(standardized z).
    z: (n,)
    returns: (n,)
    """
    z = z.astype(float, copy=True)
    z = (z - z.mean()) / (z.std() + 1e-6)
    u = 1.0 / (1.0 + np.exp(-z))  # sigmoid

    t = np.linspace(0.0, 1.0, K)
    y = np.zeros((K,), dtype=float)
    step_scale = float(rng.uniform(0.3, 1.0))
    for k in range(1, K):
        y[k] = y[k - 1] + step_scale * float(rng.normal(0.0, 1.0))
    y = y / (y.std() + 1e-6) * float(rng.uniform(0.7, 1.5))

    m = np.zeros((K,), dtype=float)
    dt = t[1] - t[0]
    m[1 : K - 1] = (y[2:] - y[:-2]) / (2 * dt)
    m[0] = (y[1] - y[0]) / dt
    m[-1] = (y[-1] - y[-2]) / dt

    idx = np.searchsorted(t, u, side="right") - 1
    idx = np.clip(idx, 0, K - 2)

    t0 = t[idx]
    t1 = t[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    h = t1 - t0
    s = (u - t0) / h

    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2

    f = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    f = f - f.mean()
    f = f * (float(rng.uniform(0.5, 1.0)) / (f.std() + 1e-6))
    return f


def apply_mechanism(funform, parents, rng, mech_mode="additive"):
    """parents: (n, P). funform in {lin,  gp, spline}
    mech_mode in {scalar, additive, non_additive}
    """
    if funform in ("lin"):
        z = parents[:, 0] if parents.shape[1] > 0 else np.zeros((parents.shape[0],))
        if funform == "lin":
            return z

    if funform == "gp":
        if mech_mode == "additive":
            out = np.zeros((parents.shape[0],), dtype=float)
            for j in range(parents.shape[1]):
                out = out + _gp_rff(parents[:, j:j+1], rng)
            out = out / np.sqrt(max(parents.shape[1], 1))
            return out
        # non_additive: GP over joint parents
        return _gp_rff(parents, rng)

    if funform == "spline":
        if mech_mode == "scalar":
            z = parents[:, 0] if parents.shape[1] > 0 else np.zeros((parents.shape[0],))
            K = int(rng.integers(4, 8))
            return _random_cubic_spline_1d(z, rng, K=K)
        # additive spline over parents
        out = np.zeros((parents.shape[0],), dtype=float)
        for j in range(parents.shape[1]):
            K = int(rng.integers(4, 8))
            out = out + _random_cubic_spline_1d(parents[:, j], rng, K=K)
        out = out / np.sqrt(max(parents.shape[1], 1))
        return out

    raise ValueError(f"Unknown funform: {funform}")


def simulate_chain_intervention(
    *,
    n_samples,
    ctx_N,
    seed,
    funform,
    noise_scale=0.1,

    mech_mode="additive",
    intervention_type="hard",
    intervene_on_train=False,
    intervene_frac=0.5,
    return_fxy=False,
):
    # Separate RNGs so mechanism parameters are fixed across splits
    rng_mech = np.random.default_rng(seed)
    rng_data = np.random.default_rng(seed + 1337)

    # Fixed weights + fixed mechanism parameters for X->Y and Y->Z

    def make_mech(fun, rng):
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        def mech(parents):
            return apply_mechanism(fun, parents, local_rng, mech_mode=mech_mode)
        return mech

    f_xy = make_mech(funform, rng_mech)
    f_yz = make_mech(funform, rng_mech)
    f_yz_alt = make_mech(funform, rng_mech)

    def gen_split(n, intervene_mask):
        x0 = rng_data.normal(0.0, noise_scale, size=(n,))
        y = f_xy((x0).reshape(-1, 1))
        y_noisefree = y.copy()
        y = y+ rng_data.normal(0.0, noise_scale, size=(n,))

        # start with non-intervened mechanism
        z = f_yz(( y).reshape(-1, 1)) + rng_data.normal(0.0, noise_scale, size=(n,))

        if intervene_mask.any():
            idx = np.where(intervene_mask)[0]
            if intervention_type == "hard":
                z[idx] = rng_data.normal(0.0, noise_scale, size=(len(idx),))
            elif intervention_type == "mech_shift":
                if f_yz_alt is None:
                    raise ValueError("intervention_funform must be set for mech_shift")
                z[idx] = f_yz_alt(( y[idx]).reshape(-1, 1)) + rng_data.normal(0.0, noise_scale, size=(len(idx),))
            else:
                raise ValueError(f"Unknown intervention_type: {intervention_type}")

        X = np.column_stack([x0, y, z])
        return X, y_noisefree

    # context with mixed interventions
    is_intervened = rng_data.random(n_samples) < float(intervene_frac)
    # dont intervene in train
    if not intervene_on_train:
        is_intervened[:ctx_N] = False
    X, y_noisefree = gen_split(n_samples, is_intervened)



    # ---- normalize using context stats ----
    ctx_mean = X[:ctx_N].mean(axis=0, keepdims=True)
    ctx_std = X[:ctx_N].std(axis=0, keepdims=True) + 1e-6
    X = (X - ctx_mean) / ctx_std
    y_noisefree = (y_noisefree - ctx_mean[0, 1]) / ctx_std[0, 1]
    if return_fxy:
        return X, is_intervened, y_noisefree
    return X, is_intervened


def synthetic_order_experiment(model, *, seeds=range(5), n_rows=128, n_variables=5):
    """Evaluate inferred and random orders on identical synthetic tables."""
    import pandas as pd
    import torch
    from taborder import SyntheticDataset, topological_divergence

    rows = []
    device = next(model.parameters()).device
    for seed in seeds:
        for masking in (0.0, 0.2, 0.4):
            torch.manual_seed(seed)
            dataset = SyntheticDataset(
                variable_mode="additive-gp", variable_range=(n_variables, n_variables),
                parent_range=(1, 3), noise_var_range=(0.1, 0.5), batch_size=1,
                min_seq_len=n_rows, max_seq_len=n_rows, seed=seed,
                masking_percentage=masking,
            )
            x, rp, cp, _, qi, dags, *_ = next(iter(dataset))
            with torch.inference_mode():
                outputs = model(x.to(device), row_mask=rp.to(device),
                                col_mask=cp.to(device), query_indicator=qi.to(device))
            score = outputs[2][0].cpu()
            random = np.mean([topological_divergence(torch.rand(n_variables), dags[0]) for _ in range(32)])
            for method, divergence in [("TabOrder", topological_divergence(score, dags[0])), ("Random order", random)]:
                rows.append(dict(seed=seed, masking=masking, method=method, divergence=divergence))
    return pd.DataFrame(rows)


def imputation_experiment(models, *, seeds=range(3), n_context=128, n_query=64):
    """Synthetic GP and offline real-table demo; evaluate hidden QUERY cells only.

    The diabetes example uses scikit-learn's bundled feature table, without its
    regression target. It is not a reproduction of CC18/CTR23 aggregate ranks.
    """
    import pandas as pd
    import torch
    from sklearn.datasets import load_diabetes
    from sklearn.impute import KNNImputer
    from taborder import SyntheticDataset, encode_table, impute_table

    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        helper = SyntheticDataset(variable_mode="additive-gp", parent_range=(1, 3), noise_var_range=(0.1, 0.5))
        synthetic, _, _ = helper.generate_dataset(n_rows=n_context + n_query, n_vars=5, seed=seed)
        real = load_diabetes().data
        real = real[rng.permutation(len(real))[:n_context + n_query]]
        for dataset, values in [("Synthetic GP", synthetic.numpy()), ("Diabetes (offline real table)", real)]:
            if len(values) < n_context + n_query:
                raise ValueError("Requested more rows than the offline diabetes example provides")
            missing = rng.random(values.shape) < 0.2
            # Hide ground truth from every imputer, including the encoder.
            hidden = np.where(missing, np.nan, values)
            encoded = encode_table(hidden, missing, context_size=n_context)
            evaluation = missing.copy()
            evaluation[:n_context] = False
            normalized_truth = (values - encoded.mean) / encoded.scale
            baseline_input = (hidden - encoded.mean) / encoded.scale
            predictions = {"Context mean": np.zeros_like(values)}
            knn = KNNImputer(n_neighbors=5, weights="distance")
            knn.fit(baseline_input[:n_context])
            predictions["KNN (context only)"] = knn.transform(baseline_input)
            for name, model in models.items():
                result = impute_table(model, hidden, missing, context_size=n_context)
                predictions[name] = (result["imputed"] - encoded.mean) / encoded.scale
            for method, predicted in predictions.items():
                rmse = np.sqrt(np.mean((predicted[evaluation] - normalized_truth[evaluation]) ** 2))
                rows.append(dict(dataset=dataset, seed=seed, method=method, normalized_rmse=float(rmse),
                                 evaluated_cells=int(evaluation.sum())))
    return pd.DataFrame(rows)


def intervention_experiment(model, *, seed=20, n_context=128, n_query=128, run_tabpfn=False, tabpfn_model_path=None):
    """Fresh paired intervention: context fixed; only query Z changes.

    Freeze an order inferred from observational context, without using the known
    chain order as a model input. Report the learned order even if it is wrong.
    The original paper used a different checkpoint and a mixed-regime query set.
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from taborder import impute_table

    kwargs = dict(n_samples=n_context + n_query, ctx_N=n_context, seed=seed,
                  funform="spline", noise_scale=0.1, mech_mode="additive",
                  intervention_type="hard", return_fxy=True)
    observational, _, noisefree = simulate_chain_intervention(**kwargs, intervene_frac=0.0)
    intervened, _, _ = simulate_chain_intervention(**kwargs, intervene_frac=1.0)
    np.testing.assert_array_equal(observational[:, :2], intervened[:, :2])
    np.testing.assert_array_equal(observational[:n_context], intervened[:n_context])
    context = observational[:n_context]
    context_scores = impute_table(model, context, np.zeros_like(context, dtype=bool),
                                  context_size=n_context)["order_scores"]
    order = np.argsort(context_scores).tolist()
    features = [0, 2]
    forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1)
    forest.fit(context[:, features], context[:, 1])
    baselines = {"Random forest": forest}
    if run_tabpfn:
        # Optional: may download weights or require provider authentication.
        from tabpfn import TabPFNRegressor
        options = dict(device=str(next(model.parameters()).device), random_state=seed)
        if tabpfn_model_path is not None:
            options["model_path"] = str(tabpfn_model_path)
        tabpfn = TabPFNRegressor(**options)
        tabpfn.fit(context[:, features], context[:, 1])
        baselines["TabPFN (fresh run)"] = tabpfn
    predictions, metrics = [], []
    for condition, values in [("Observational", observational), ("Intervened", intervened)]:
        missing = np.zeros(values.shape, dtype=bool)
        missing[n_context:, 1] = True
        hidden = np.where(missing, np.nan, values)
        result = impute_table(model, hidden, missing, context_size=n_context,
                              order_scores_override=context_scores[None, :])
        predicted = {"TabOrder (context-learned order)": result["predicted_mean"][n_context:, 1]}
        for name, estimator in baselines.items():
            predicted[name] = estimator.predict(values[n_context:, features])
        for name, mean in predicted.items():
            mse = float(np.mean((mean - values[n_context:, 1]) ** 2))
            metrics.append(dict(method=name, condition=condition, mse=mse))
            for i in range(n_query):
                predictions.append(dict(method=name, condition=condition, sample=i,
                                        x=values[n_context + i, 0], y_true=values[n_context + i, 1],
                                        y_noisefree=noisefree[n_context + i], predicted_y=float(mean[i])))
    return pd.DataFrame(predictions), pd.DataFrame(metrics), order


def plot_paper_figure1(predictions):
    """Reconstruct the paper's two panels from the saved prediction table."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True, sharey=True)
    intervened = predictions["is_intervened"].astype(bool)
    ordered = predictions.sort_values("x")
    for ax, column, title in zip(axes, ["y_pred_tabpfn", "y_pred_cfm"], ["(a) TabPFN", "(b) TabOrder"]):
        for selected, color, label in [(~intervened, "#85bb65", "Observational"), (intervened, "#f28e2b", "Intervened")]:
            ax.scatter(predictions.loc[selected, "x"], predictions.loc[selected, column],
                       s=8, alpha=0.3, color=color, label=label)
        ax.plot(ordered["x"], ordered["y_noisefree"], color="black", linewidth=2, label="$f(X)$")
        ax.set(title=title, xlabel="$X$ (normalized)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted $Y$ (normalized)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig
