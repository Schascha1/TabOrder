"""Synthetic SCM prior used by the additive and non-additive GP training runs.

All generated variables are continuous. Set both torch.manual_seed and the
dataset seed for reproducible samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

@dataclass
class VariableSpec:
    var_type: str  # "continuous" or "categorical"
    num_categories: int = 0


class SyntheticDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        device: str = "cpu",
        variable_range: Tuple[int, int] = (5, 20),
        subdag_range: Tuple[int, int] = None,
        parent_range: Tuple[int, int] = (0, 4),
        max_out_degree: int = 4,
        noise_var_range: Tuple[float, float] = (0.05, 0.5),
        variable_mode: str = "non-additive-gp",
        discrete_prob: float = 0.25,
        binary_prob: float = 0.25,
        max_categories: int = 10,
        hidden_dim_range: Tuple[int, int] = (6, 16),
        batch_size: int = 4,
        min_seq_len: int = 50,
        max_seq_len: int = 1024,
        min_query_frac: float = 0.05,
        max_query_frac: float = 0.5,
        masking_percentage: float = 0.2,
        masking_pattern: str = "MCAR",
        mar_driver_col: int = 0,
        mar_alpha: float = 2.0,
        mnar_alpha: float = 2.0,
        predictive_task: bool = False,
        target_mode: str = "random_with_parents",
        normalize: str = "context",
        n_vars: Optional[int] = None,
        seed: Optional[int] = None,
        use_heteroscedastic_noise: bool = False,
        mechanism_range: Tuple[int, int] = (1, 3),
        noise_mode: str = "additive",
    ):
        super().__init__()
        self.device = device
        self.variable_range = variable_range
        self.parent_range = parent_range
        self.max_out_degree = max_out_degree
        self.noise_var_range = noise_var_range
        self.variable_mode = variable_mode
        if variable_mode not in ("additive-gp", "non-additive-gp", "additive-poly", "mixed"):
            raise ValueError(
                "variable_mode must be 'additive-gp', 'non-additive-gp', 'additive-poly', or 'mixed'"
            )

        self.discrete_prob = discrete_prob
        self.binary_prob = binary_prob
        self.max_categories = max_categories
        self.hidden_dim_range = hidden_dim_range
        self.batch_size = batch_size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_query_frac = min_query_frac
        self.max_query_frac = max_query_frac
        self.masking_percentage = masking_percentage
        self.masking_pattern = masking_pattern
        self.mar_driver_col = mar_driver_col
        self.mar_alpha = mar_alpha
        self.mnar_alpha = mnar_alpha
        self.predictive_task = predictive_task
        self.target_mode = target_mode
        self.normalize = normalize
        self.n_vars = n_vars
        self.seed = seed
        self._iter_count = 0
        self.use_heteroscedastic_noise = use_heteroscedastic_noise
        self.mechanism_range = mechanism_range

        if noise_mode not in ("additive", "heteroskedastic", "multiplicative", "location_scale"):
            raise ValueError(
                "noise_mode must be 'additive', 'heteroskedastic', 'multiplicative', or 'location_scale'"
            )

        # Backward compatibility: old configs that only toggle heteroscedastic noise
        if use_heteroscedastic_noise and noise_mode == "additive":
            self.noise_mode = "heteroskedastic"
        else:
            self.noise_mode = noise_mode

        if subdag_range is None:
            self.subdag_range = variable_range
        else:
            self.subdag_range = subdag_range

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        base_seed = self.seed
        while True:
            seed = None
            if base_seed is not None:
                seed = base_seed + self._iter_count + worker_id * 100000
            self._iter_count += 1
            yield self.generate_batch_with_masks(
                batch_size=self.batch_size,
                min_seq_len=self.min_seq_len,
                max_seq_len=self.max_seq_len,
                min_query_frac=self.min_query_frac,
                max_query_frac=self.max_query_frac,
                masking_percentage=self.masking_percentage,
                masking_pattern=self.masking_pattern,
                mar_driver_col=self.mar_driver_col,
                mar_alpha=self.mar_alpha,
                mnar_alpha=self.mnar_alpha,
                predictive_task=self.predictive_task,
                target_mode=self.target_mode,
                normalize=self.normalize,
                n_vars=self.n_vars,
                seed=seed,
            )

    def sample_global_dag(self, n_vars: int, rng: np.random.Generator) -> Tuple[nx.DiGraph, List[int]]:
        """Sample a DAG."""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n_vars))
        n_vars_remaining = n_vars
        dag, _ = self._sample_local_dag(n_vars, rng)
        # min_subdag_size = min(self.subdag_range[0], n_vars_remaining)
        # max_subdag_size = self.subdag_range[1] if self.subdag_range[1] < n_vars_remaining else n_vars_remaining
        # while n_vars_remaining > 0:
        #     subdag_size = int(rng.integers(min_subdag_size, max_subdag_size + 1))
        #     subdag, topo = self._sample_local_dag(subdag_size, rng)
        #     mapping = {i: n_vars - n_vars_remaining + i for i in range(subdag_size)}
        #     subdag = nx.relabel_nodes(subdag, mapping)
        #     dag = nx.compose(dag, subdag)
        #     n_vars_remaining -= subdag_size
        #     min_subdag_size = min(self.subdag_range[0], n_vars_remaining)
        #     max_subdag_size = self.subdag_range[1] if self.subdag_range[1] < n_vars_remaining else n_vars_remaining
        topological_order = list(nx.topological_sort(dag))
        return dag, topological_order

    def _sample_local_dag(self, n_vars: int, rng: np.random.Generator) -> Tuple[nx.DiGraph, List[int]]:
        """Sample a DAG."""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(n_vars))
        topological_order = list(range(n_vars))
        rng.shuffle(topological_order)
        full_nodes: List[int] = []
        for idx, node in enumerate(topological_order):
            if idx == 0:
                continue
            eligible_parents = [p for p in topological_order[:idx] if p not in full_nodes]
            max_possible = min(len(eligible_parents), self.parent_range[1], idx)
            min_possible = min(self.parent_range[0], max_possible)
            n_parents = rng.integers(min_possible, max_possible + 1) if max_possible > 0 else 0
            parents = rng.choice(eligible_parents, n_parents, replace=False) if n_parents > 0 else []
            for parent in parents:
                dag.add_edge(parent, node)
                if dag.out_degree(parent) >= self.max_out_degree:
                    full_nodes.append(parent)
        return dag, topological_order

    def sample_variable_specs(self, n_vars: int, rng: np.random.Generator) -> List[VariableSpec]:
        specs = []
        for _ in range(n_vars):
            specs.append(VariableSpec(var_type="continuous", num_categories=0))
        return specs

    def _sample_exogenous(self, n_rows: int, rng: np.random.Generator) -> torch.Tensor:
        """Sample a single exogenous variable with mixed distributions."""
        source = rng.choice(["normal", "uniform"])
        if source == "normal":
            x = torch.randn(n_rows, device=self.device)
            x = x * float(rng.uniform(0.5, 2.0))
        elif source == "uniform":
            x = torch.rand(n_rows, device=self.device)
            x = (x - 0.5) * float(rng.uniform(1.0, 5.0))
        m = x.mean()
        s = x.std().clamp_min(1e-6)
        x = torch.clamp(x, m - 6 * s, m + 6 * s)
        return x

    def _eta_from_parents(self, parent_data: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        x = parent_data  # (n_rows, P)
        n_rows, P = x.shape
        hidden_dim = 256

        # Random Fourier features for an RBF GP prior over the mechanism.
        x = x - x.mean(dim=0, keepdim=True)
        x = x / (x.std(dim=0, keepdim=True) + 1e-6)
        lengthscale = float(np.exp(rng.uniform(np.log(0.3), np.log(1.0))))

        w = torch.randn(P, hidden_dim, device=self.device) / lengthscale
        b = torch.rand(hidden_dim, device=self.device) * (2.0 * math.pi)
        phi = math.sqrt(2.0 / hidden_dim) * torch.cos(x @ w + b)
        weights = torch.randn(hidden_dim, device=self.device)
        eta = phi @ weights

        return eta

    def _parent_dependent_scale_and_shift(
        self,
        parent_data: torch.Tensor,
        rng: np.random.Generator,
        base_sigma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns parent-dependent noise scale and shift.

        Returns:
            sigma: (n_rows,) positive scale
            loc:   (n_rows,) mean/shift term
        """
        n_rows, P = parent_data.shape
        device = parent_data.device

        if P == 0:
            sigma = torch.full((n_rows,), float(base_sigma), device=device)
            loc = torch.zeros(n_rows, device=device)
            return sigma, loc

        z = parent_data
        z = z - z.mean(dim=0, keepdim=True)
        z = z / (z.std(dim=0, keepdim=True) + 1e-6)

        w_scale = torch.randn(P, device=device) / math.sqrt(P)
        w_loc = torch.randn(P, device=device) / math.sqrt(P)

        raw_scale = z @ w_scale
        sigma = base_sigma * (0.25 + 1.25 * torch.sigmoid(raw_scale))

        loc = 0.3 * (z @ w_loc)
        return sigma, loc

    def _continuous_from_parents(
        self,
        parent_data: torch.Tensor,
        parent_specs: List[VariableSpec],
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """
        Strong continuous mechanism:
        - sum of several nonlinear mechanisms, each using a subset of parents
        - constructed so that every parent is used in at least one mechanism
        """
        device = parent_data.device
        n_rows, P = parent_data.shape

        if P == 0:
            # no parents: pure exogenous-like noise
            base_sigma = float(rng.uniform(0.5, 1.5))
            x = torch.randn(n_rows, device=device) * base_sigma
            m = x.mean()
            s = x.std().clamp_min(1e-6)
            x = torch.clamp(x, m - 6 * s, m + 6 * s)
            return x

        if self.variable_mode == "additive-gp":
            effect = 0
            for j in range(parent_data.shape[1]):
                effect = effect + self._eta_from_parents(parent_data[:, j:j+1], rng)

        elif self.variable_mode == "additive-poly":
            effect = 0
            fan_in = max(P, 1)
            effect = torch.zeros(n_rows, device=device)

            # per-parent polynomial ANM: sum_j f_j(z_j)
            for j in range(P):
                random_cubic_spline = self.random_cubic_spline(
                    parent_data[:, j], rng, K=int(rng.integers(4, 8))
                )
                effect = effect + random_cubic_spline / math.sqrt(fan_in)

        elif self.variable_mode == "non-additive-gp":
            effect = self._eta_from_parents(parent_data, rng)

        elif self.variable_mode == "mixed":
            if rng.random() < 0.33:
                effect = self._eta_from_parents(parent_data, rng)
            else:
                effect = 0
                for j in range(parent_data.shape[1]):
                    effect = effect + self._eta_from_parents(parent_data[:, j:j+1], rng)

        else:
            raise ValueError(f"Unknown variable_mode: {self.variable_mode}")

        # ---- noise model ----
        base_sigma = float(rng.uniform(self.noise_var_range[0], self.noise_var_range[1]))
        eps = torch.randn_like(effect)

        if self.noise_mode == "additive":
            sigma = base_sigma
            noise = sigma * eps
            x = effect + noise

        elif self.noise_mode == "heteroskedastic":
            sigma, _ = self._parent_dependent_scale_and_shift(parent_data, rng, base_sigma)
            noise = sigma * eps
            x = effect + noise

        elif self.noise_mode == "multiplicative":
            sigma = base_sigma
            mult = 1.0 + sigma * eps
            x = effect * mult

        elif self.noise_mode == "location_scale":
            sigma, loc = self._parent_dependent_scale_and_shift(parent_data, rng, base_sigma)
            x = effect + loc + sigma * eps

        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        # light guardrail: clip extremes, no forced unit variance
        m = x.mean()
        s = x.std().clamp_min(1e-6)
        x = torch.clamp(x, m - 6 * s, m + 6 * s)
        return x

    def random_cubic_spline(
        self,
        z: torch.Tensor,
        rng: np.random.Generator,
        K: int = 6,
    ) -> torch.Tensor:
        """
        Returns f(z) where f is a random smooth cubic spline.
        No clamping. Fully smooth. Good for nonlinear ANMs.

        z: Tensor (n,)
        K: number of spline knots
        """
        device = z.device

        # 1. Standardize z
        z = (z - z.mean()) / z.std().clamp_min(1e-6)

        # 2. Map ℝ -> (0,1) smoothly
        u = torch.sigmoid(z)

        # 3. Random knot positions
        t = torch.linspace(0.0, 1.0, K, device=device)

        # 4. Random knot values with smoothness
        y = torch.zeros(K, device=device)
        step_scale = float(rng.uniform(0.3, 1.0))
        for k in range(1, K):
            y[k] = y[k - 1] + step_scale * float(rng.normal(0.0, 1.0))

        y = y / (y.std().clamp_min(1e-6)) * float(rng.uniform(0.7, 1.5))

        # 5. Slopes at knots
        m = torch.zeros(K, device=device)
        dt = t[1] - t[0]
        m[1:K - 1] = (y[2:] - y[:-2]) / (2 * dt)
        m[0] = (y[1] - y[0]) / dt
        m[-1] = (y[-1] - y[-2]) / dt

        # 6. Evaluate cubic Hermite spline at u
        idx = torch.bucketize(u, t) - 1
        idx = idx.clamp(0, K - 2)

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

        f = (h00 * y0 +
             h10 * h * m0 +
             h01 * y1 +
             h11 * h * m1)

        # 7. Normalize spline output scale
        f = f - f.mean()
        f_std = f.std().clamp_min(1e-6)
        target_std = float(rng.uniform(0.5, 1.0))
        f = f * (target_std / f_std)

        return f

    def generate_dataset(
        self,
        n_rows: int,
        n_vars: Optional[int] = None,
        seed: Optional[int] = None,
        dag: Optional[nx.DiGraph] = None,
        variable_specs: Optional[List[VariableSpec]] = None,
    ) -> Tuple[torch.Tensor, nx.DiGraph, List[VariableSpec]]:
        """
        Generate a single dataset with a global DAG and mixed variable types.

        Returns:
            X: (n_rows, n_vars) tensor
            dag: networkx.DiGraph
            variable_specs: list describing variable types/categories
        """
        rng = np.random.default_rng(seed)
        if n_vars is None:
            n_vars = int(rng.integers(self.variable_range[0], self.variable_range[1] + 1))

        if dag is None:
            dag, topo = self.sample_global_dag(n_vars, rng)
        else:
            topo = list(nx.topological_sort(dag))

        if variable_specs is None:
            variable_specs = self.sample_variable_specs(n_vars, rng)

        X = torch.zeros((n_rows, n_vars), device=self.device)
        for node in topo:
            parents = list(dag.predecessors(node))
            spec = variable_specs[node]
            if len(parents) == 0:
                X[:, node] = self._sample_exogenous(n_rows, rng)
                continue

            parent_nodes = parents
            parent_data = X[:, parent_nodes]
            parent_specs = [variable_specs[p] for p in parent_nodes]
            X[:, node] = self._continuous_from_parents(parent_data, parent_specs, rng)

        return X, dag, variable_specs

    def generate_batch(
        self,
        batch_size: int,
        n_rows: int,
        n_vars: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Generate a batch using a single global DAG and shared variable specs.

        Returns a dict with:
            X: (B, n_rows, n_vars)
            dag: global DAG
            variable_specs: list of VariableSpec
        """
        rng = np.random.default_rng(seed)
        if n_vars is None:
            n_vars = int(rng.integers(self.variable_range[0], self.variable_range[1] + 1))
        dag, _ = self.sample_global_dag(n_vars, rng)
        variable_specs = self.sample_variable_specs(n_vars, rng)

        X_list = []
        for _ in range(batch_size):
            X, _, _ = self.generate_dataset(
                n_rows=n_rows,
                n_vars=n_vars,
                seed=int(rng.integers(0, 2**32 - 1)),
                dag=dag,
                variable_specs=variable_specs,
            )
            X_list.append(X)

        X_batch = torch.stack(X_list, dim=0)
        return {"X": X_batch, "dag": dag, "variable_specs": variable_specs}

    @staticmethod
    def _causal_relation_indicator(
        dag: nx.DiGraph, n_vars: int, device: str
    ) -> torch.Tensor:
        indicator = torch.zeros((n_vars, n_vars), device=device, dtype=torch.bool)
        for node in dag.nodes():
            descendants = nx.descendants(dag, node)
            for desc in descendants:
                indicator[node, desc] = True
        return indicator

    @staticmethod
    def _select_target_node(
        dag: nx.DiGraph, rng: np.random.Generator, mode: str
    ) -> int:
        if mode == "random_with_parents":
            candidates = [n for n in dag.nodes() if dag.in_degree(n) > 0]
        elif mode == "sink":
            candidates = [n for n in dag.nodes() if dag.out_degree(n) == 0]
        else:
            raise ValueError(f"Unknown target_mode: {mode}")
        if not candidates:
            candidates = list(dag.nodes())
        return int(rng.choice(candidates))

    def generate_batch_with_masks(
        self,
        batch_size: int,
        min_seq_len: int = 50,
        max_seq_len: int = 1024,
        min_query_frac: float = 0.05,
        max_query_frac: float = 0.5,
        masking_percentage: float = 0.2,
        masking_pattern: str = "MCAR",
        mar_driver_col: int = 0,
        mar_alpha: float = 2.0,
        mnar_alpha: float = 2.0,
        predictive_task: bool = False,
        target_mode: str = "random_with_parents",
        normalize: str = "context",
        n_vars: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[nx.DiGraph],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Generate a masked batch in the same format as SyntheticContinuousDataset.

        Returns:
            batch: (B, N, D, 4) values + mask indicator + categorical flag + max_categories
            row_padding_mask: (B, N)
            col_padding_mask: (B, D)
            unmasked_batch: (B, N, D)
            query_indicator: (B, N)
            dags: list of DAGs (global DAG repeated)
            masking_indicator: (B, N, D)
            causal_relation_indicator: (B, D, D)
            not_padded_indicator: (B, N, D)
            predictive_parent_availability: (B, N, D)
            target_indices: (B,) or None
            column_type_indicator: (B, D)
        """
        rng = np.random.default_rng(seed)
        if min_seq_len < 1 or max_seq_len < min_seq_len:
            raise ValueError("min_seq_len must be >= 1 and <= max_seq_len")
        if not (0.0 < min_query_frac <= max_query_frac < 1.0):
            raise ValueError("min_query_frac and max_query_frac must be in (0, 1)")

        batch_seq_lens = rng.integers(min_seq_len, max_seq_len + 1, size=batch_size)
        batch_query_fracs = rng.uniform(min_query_frac, max_query_frac, size=batch_size)
        batch_query_sizes = np.maximum(1, np.floor(batch_seq_lens * batch_query_fracs).astype(int))
        batch_context_sizes = batch_seq_lens - batch_query_sizes
        n_rows_batch = int(batch_seq_lens.max())

        if n_vars is None:
            n_vars = int(rng.integers(self.variable_range[0], self.variable_range[1] + 1))
        dag, _ = self.sample_global_dag(n_vars, rng)
        variable_specs = self.sample_variable_specs(n_vars, rng)

        batch = torch.zeros((batch_size, n_rows_batch, n_vars, 4), device=self.device)
        row_padding_mask = torch.zeros((batch_size, n_rows_batch), device=self.device, dtype=torch.bool)
        col_padding_mask = torch.zeros((batch_size, n_vars), device=self.device, dtype=torch.bool)
        query_indicator = torch.zeros((batch_size, n_rows_batch), device=self.device, dtype=torch.bool)
        masking_indicator = torch.zeros((batch_size, n_rows_batch, n_vars), device=self.device, dtype=torch.bool)
        not_padded_indicator = torch.zeros((batch_size, n_rows_batch, n_vars), device=self.device, dtype=torch.bool)
        context_means = torch.zeros((batch_size, n_vars), device=self.device)

        causal_relation_indicator = self._causal_relation_indicator(dag, n_vars, self.device)
        causal_relation_indicator = causal_relation_indicator.unsqueeze(0).repeat(batch_size, 1, 1)

        target_indices = torch.full((batch_size,), -1, device=self.device, dtype=torch.long) if predictive_task else None

        feature_type = torch.zeros((batch_size, n_rows_batch, n_vars), device=self.device, dtype=torch.float)
        max_categories = torch.zeros((batch_size, n_rows_batch, n_vars), device=self.device, dtype=torch.float)
        for j, spec in enumerate(variable_specs):
            if spec.var_type == "categorical":
                feature_type[:, :, j] = 1.0
                max_categories[:, :, j] = float(spec.num_categories)

        column_type_indicator = feature_type[:, 0, :].clone()

        for i in range(batch_size):
            context_size = int(batch_context_sizes[i])
            n_rows = int(batch_seq_lens[i])

            X, _, _ = self.generate_dataset(
                n_rows=n_rows,
                n_vars=n_vars,
                seed=int(rng.integers(0, 2**32 - 1)),
                dag=dag,
                variable_specs=variable_specs,
            )
            batch[i, :n_rows, :, 0] = X
            row_padding_mask[i, n_rows:] = True
            query_indicator[i, context_size:n_rows] = True
            not_padded_indicator[i, :n_rows, :] = True

            if predictive_task:
                target_idx = self._select_target_node(dag, rng, target_mode)
                target_indices[i] = target_idx
                masking_indicator[i, :n_rows, target_idx] = True
            else:
                if masking_pattern not in ("MCAR", "MAR", "MNAR"):
                    raise ValueError("masking_pattern must be 'MCAR', 'MAR', or 'MNAR'")
                if masking_pattern == "MCAR":
                    mask = torch.rand((n_rows, n_vars), device=self.device) < masking_percentage
                elif masking_pattern == "MAR":
                    driver = min(max(int(mar_driver_col), 0), n_vars - 1)
                    x = batch[i, :n_rows, driver, 0]
                    x = (x - x.mean()) / (x.std() + 1e-6)
                    prob = torch.sigmoid(mar_alpha * x) * masking_percentage * 2.0
                    prob = prob.clamp(0.0, 0.9).unsqueeze(1).expand(n_rows, n_vars)
                    mask = torch.rand((n_rows, n_vars), device=self.device) < prob
                else:
                    x = batch[i, :n_rows, :, 0]
                    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
                    prob = torch.sigmoid(mnar_alpha * x) * masking_percentage * 2.0
                    prob = prob.clamp(0.0, 0.9)
                    mask = torch.rand((n_rows, n_vars), device=self.device) < prob
                masking_indicator[i, :n_rows, :] = mask

            if normalize not in ("context", "none"):
                raise ValueError("normalize must be 'context' or 'none'")
            if normalize == "context":
                context_rows = torch.zeros((n_rows_batch,), device=self.device, dtype=torch.bool)
                context_rows[:context_size] = True
                valid_rows = context_rows & (~row_padding_mask[i])
                continuous_cols = feature_type[i, 0, :] == 0.0
                for col_idx in torch.where(continuous_cols)[0]:
                    col_mask = valid_rows & (~masking_indicator[i, :, col_idx])
                    if col_mask.any():
                        vals = batch[i, col_mask, col_idx, 0]
                        mean = vals.mean()
                        std = vals.std().clamp_min(1e-6)
                        batch[i, :n_rows, col_idx, 0] = (batch[i, :n_rows, col_idx, 0] - mean) / std

            context_means[i, :] = batch[i, :context_size, :, 0].mean(dim=0)

        predictive_parent_availability = torch.zeros(
            (batch_size, n_rows_batch, n_vars), device=self.device, dtype=torch.long
        )
        for i in range(batch_size):
            for node in dag.nodes():
                parents = list(dag.predecessors(node))
                all_parents_available = torch.ones((n_rows_batch), device=self.device, dtype=torch.bool)
                no_parents_available = torch.ones((n_rows_batch), device=self.device, dtype=torch.bool) * (len(parents) > 0)
                for parent in parents:
                    parent_masking_ind = masking_indicator[i, :, parent]
                    all_parents_available = all_parents_available & (~parent_masking_ind)
                    no_parents_available = no_parents_available & parent_masking_ind
                predictive_parent_availability[i, :, node] = (
                    (1 - all_parents_available.long()) + no_parents_available.long()
                )

        unmasked_batch = batch[:, :, :, 0].detach().clone()
        context_means = context_means.unsqueeze(1)
        mask_values = masking_indicator.float() * context_means
        batch[:, :, :, 0] = batch[:, :, :, 0] * (~masking_indicator).float() + mask_values
        batch[:, :, :, 1] = masking_indicator.float()
        batch[:, :, :, 2] = feature_type
        batch[:, :, :, 3] = max_categories

        dags = [dag for _ in range(batch_size)]
        return (
            batch,
            row_padding_mask,
            col_padding_mask,
            unmasked_batch,
            query_indicator,
            dags,
            masking_indicator,
            causal_relation_indicator,
            not_padded_indicator,
            predictive_parent_availability,
            target_indices,
            column_type_indicator,
        )
