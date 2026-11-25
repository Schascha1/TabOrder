"""Model variant used by the CC18/CTR23 fine-tuning notebook."""

from typing import Optional, Literal

import torch
from torch import nn


class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, num_layers=1, num_heads=8, d_ff=None, dropout=0.1, prenorm=True):
        super().__init__()
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, num_heads,
                dim_feedforward=d_ff or 2 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=prenorm
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=attn_mask, src_key_padding_mask=key_padding_mask)
        return x


class ParentAttentionHead(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 1, forbid_self: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.2,
            batch_first=True,
        )
        self.forbid_self = forbid_self

    def forward(
        self,
        h: torch.Tensor,                 # (B, C, D)
        col_mask: torch.Tensor,          # (B, C) True=pad
        attn_mask: Optional[torch.Tensor] = None  # (C,C) or (B*H,C,C)
    ) -> torch.Tensor:
        B, C, D = h.shape
        H = self.attn.num_heads

        # Normalize attn_mask into a valid shape for MultiheadAttention
        # Allowed: (C,C) or (B*H,C,C)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (C,C) ok
                pass
            elif attn_mask.dim() == 3:
                # Could be (B,C,C) from upstream; expand to (B*H,C,C)
                if attn_mask.shape[0] == B:
                    attn_mask = attn_mask.repeat_interleave(H, dim=0)  # (B*H,C,C)
                # If it is already (B*H,C,C), keep as-is
            else:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.dim()}D")

        # Forbid self with an additive -inf diagonal (broadcast-safe, no weird indexing)
        if self.forbid_self:
            diag = torch.eye(C, device=h.device, dtype=torch.bool)
            if attn_mask is None:
                attn_mask = torch.zeros((C, C), device=h.device)
                attn_mask = attn_mask.masked_fill(diag, float("-inf"))
            else:
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.masked_fill(diag, float("-inf"))
                else:
                    # (B*H, C, C)
                    attn_mask = attn_mask.masked_fill(diag.unsqueeze(0), float("-inf"))

        _, attn_weights = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=col_mask,   # True = ignore as key/value
            attn_mask=attn_mask,
            need_weights=True,
        )
        return attn_weights


def masked_mean_std(x: torch.Tensor, valid: torch.Tensor, dim: int, eps: float = 1e-3):
    """
    x:     (..., C)
    valid: same shape as x (bool), True=valid
    """
    v = valid.float()
    denom = v.sum(dim=dim, keepdim=True).clamp_min(1.0)
    mean = (x * v).sum(dim=dim, keepdim=True) / denom
    var = ((x - mean) ** 2 * v).sum(dim=dim, keepdim=True) / denom
    std = var.sqrt().clamp_min(eps)
    return mean, std


class CausalOrderDAGModel(nn.Module):
    def __init__(
        self,
        d_model,
        d_classifier,
        d_embedding=128,
        K_attention_blocks_order=2,
        K_attention_blocks_pred=3,
        heads_col_sab_order=4,
        heads_row_sab_order=4,
        num_layers_col_sab_order=1,
        num_layers_row_sab_order=1,
        heads_col_sab_pred=4,
        heads_row_sab_pred=4,
        num_layers_col_sab_pred=1,
        num_layers_row_sab_pred=1,
        temperature_order=1.0,
        attention_bias=-5,
        context_col_idx: Optional[int] = None,
        order_mode: Literal["causal", "noorder"] = "causal",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.context_col_idx = context_col_idx

        self.cell_embedder = nn.Linear(d_model, d_embedding)

        self.order_col_attention_blocks = nn.ModuleList([
            SetAttentionBlock(d_embedding, num_layers=num_layers_col_sab_order, num_heads=heads_col_sab_order)
            for _ in range(K_attention_blocks_order)
        ])
        self.order_row_attention_blocks = nn.ModuleList([
            SetAttentionBlock(d_embedding, num_layers=num_layers_row_sab_order, num_heads=heads_row_sab_order)
            for _ in range(K_attention_blocks_order)
        ])

        self.pred_col_attention_blocks = nn.ModuleList([
            SetAttentionBlock(d_embedding, num_layers=num_layers_col_sab_pred, num_heads=heads_col_sab_pred)
            for _ in range(K_attention_blocks_pred)
        ])
        self.pred_row_attention_blocks = nn.ModuleList([
            SetAttentionBlock(d_embedding, num_layers=num_layers_row_sab_pred, num_heads=heads_row_sab_pred)
            for _ in range(K_attention_blocks_pred)
        ])

        self.d_embedding = d_embedding

        self.order_decoder = nn.Sequential(
            nn.Linear(d_embedding, d_embedding),
            nn.ReLU(),
            nn.Linear(d_embedding, 1)
        )

        self.discrete_cell_decoder = nn.Sequential(
            nn.Linear(d_embedding, d_classifier)
        )

        self.continuous_heteroscedastic_head = nn.Sequential(
            nn.Linear(d_embedding, d_embedding),
            nn.ReLU(),
            nn.Linear(d_embedding, 2),
        )
        self.continuous_unavailable_logit_bias = nn.Sequential(
            nn.Linear(d_embedding, d_embedding),
            nn.ReLU(),
            nn.Linear(d_embedding, 3),
        )

        self.parent_head = ParentAttentionHead(d_embedding, forbid_self=True)
        self.temperature_order = temperature_order
        self.attention_bias = attention_bias
        self.order_mode = order_mode

    def forward(self, x, col_mask=None, row_mask=None, query_indicator=None, order_scores_override: Optional[torch.Tensor] = None):
        B, N, C, D_in = x.size()

        # masks
        repeated_col_mask = col_mask.repeat_interleave(N, dim=0)  # (B*N, C)
        row_mask_bn = row_mask.repeat_interleave(C, dim=0)        # (B*C, N)

        embedded_input = self.cell_embedder(x)                    # (B, N, C, d_emb)
        current_state = embedded_input

        # ---------- order stack ----------
        for i in range(len(self.order_col_attention_blocks)):
            col_attention_order = self.order_col_attention_blocks[i]
            row_attention_order = self.order_row_attention_blocks[i]

            # batch per row: (B*N, C, D)
            row_batches = current_state.reshape(B * N, C, self.d_embedding)
            col_embeddings = col_attention_order(row_batches, key_padding_mask=repeated_col_mask)
            col_embeddings = col_embeddings.reshape(B, N, C, self.d_embedding).transpose(1, 2)  # (B, C, N, D)

            # batch per column: (B*C, N, D)
            col_embeddings = col_embeddings.reshape(B * C, N, self.d_embedding)
            col_embeddings = row_attention_order(col_embeddings, key_padding_mask=row_mask_bn)
            current_state = col_embeddings.reshape(B, C, N, self.d_embedding).transpose(1, 2)  # (B, N, C, D)

        # order scores per column
        order_scores = self.order_decoder(current_state).transpose(1, 2)  # (B, C, N, 1)

        valid_row = (~row_mask_bn).view(B, C, N, 1).float()
        order_scores = order_scores * valid_row
        order_scores = order_scores.sum(dim=2) / torch.clamp_min(valid_row.sum(dim=2), 1.0)  # (B, C, 1)
        order_scores = order_scores.squeeze(-1)  # (B, C)

        # normalize over VALID columns only
        valid_col = (~col_mask)  # (B, C)
        mean, std = masked_mean_std(order_scores, valid_col, dim=1, eps=1e-3)
        order_scores = (order_scores - mean) / std

        learned_order_scores = order_scores

        if order_scores_override is not None:
            order_scores_used = order_scores_override.to(
                device=x.device,
                dtype=learned_order_scores.dtype,
            )
        else:
            order_scores_used = learned_order_scores

        if self.order_mode == "noorder":
            p_ancestor = torch.ones((B, C, C), device=x.device, dtype=x.dtype)
            eye = torch.eye(C, device=x.device, dtype=torch.bool)[None]
            p_ancestor = p_ancestor.masked_fill(eye, 0.0)

            attn_mask = None
        else:
            # delta_ij = score_i - score_j
            delta = (
                            order_scores_used[:, :, None]
                            - order_scores_used[:, None, :]
                    ) / self.temperature_order
            p_ancestor = torch.sigmoid(delta)
            if not self.training:
                p_ancestor = (p_ancestor >= 0.5).float()

            soft_mask = self.attention_bias * (1.0 - p_ancestor)
            hard_allow = (p_ancestor >= 0.5)
            hard_mask = torch.where(hard_allow, torch.zeros_like(delta), self.attention_bias * torch.ones_like(delta))

            eye = torch.eye(C, device=delta.device, dtype=torch.bool)[None]
            soft_mask = torch.where(eye, torch.zeros_like(soft_mask), soft_mask)
            hard_mask = torch.where(eye, torch.zeros_like(hard_mask), hard_mask)

            # straight-through
            attn_mask = hard_mask.detach() + (soft_mask - soft_mask.detach())

        # ---------- parent prediction embedding h ----------
        valid_row_causal = (~row_mask).view(B, N, 1, 1).float()
        valid_col_f = (~col_mask).view(B, 1, C, 1).float()
        valid_row_causal = valid_row_causal * valid_col_f  # (B, N, C, 1)

        denom = valid_row_causal.sum(dim=1).clamp_min(1.0)  # (B, C, 1)
        h = (current_state * valid_row_causal).sum(dim=1) / denom  # (B, C, D)

        # ParentAttentionHead needs (C,C) or (B*H,C,C). We pass (B,C,C) and it will expand.
        parent_weights = self.parent_head(h, col_mask, attn_mask=attn_mask)  # (B, C, C)
        causal_edge_scores = parent_weights * p_ancestor                     # (B, C, C)

        # ---------- prediction stack: build per-row (BN*H,C,C) src_mask ----------
        if attn_mask is None:
            attn_bias_bn = None
        else:
            attn_bias_bn = attn_mask.repeat_interleave(N, dim=0)

        # CONTEXT COLUMN LOGIC (taken from impl #2)
        # CONTEXT COLUMN LOGIC (out-of-place, autograd-safe)
        if self.context_col_idx is not None and attn_bias_bn is not None:
            k = int(self.context_col_idx)

            # M is 1 everywhere except row/col k which are 0
            M = torch.ones((C, C), device=attn_bias_bn.device, dtype=attn_bias_bn.dtype)
            M[k, :] = 0.0
            M[:, k] = 0.0

            # broadcast to (B*N, C, C)
            attn_bias_bn = attn_bias_bn * M.unsqueeze(0)
        if attn_bias_bn is None:
            src_mask = None
        else:
            Hpred = self.pred_col_attention_blocks[0].num_heads
            src_mask = attn_bias_bn.repeat_interleave(Hpred, dim=0)


        current_state = embedded_input
        for i in range(len(self.pred_col_attention_blocks)):
            col_attention_pred = self.pred_col_attention_blocks[i]
            row_attention_pred = self.pred_row_attention_blocks[i]

            row_batches = current_state.reshape(B * N, C, self.d_embedding)
            col_embeddings = col_attention_pred(
                row_batches,
                key_padding_mask=repeated_col_mask,
                attn_mask=src_mask
            )
            col_embeddings = col_embeddings.reshape(B, N, C, self.d_embedding).transpose(1, 2)

            col_embeddings = col_embeddings.reshape(B * C, N, self.d_embedding)
            col_embeddings = row_attention_pred(col_embeddings, key_padding_mask=row_mask_bn)
            current_state = col_embeddings.reshape(B, C, N, self.d_embedding).transpose(1, 2)

        masking_indicator = x[:, :, :, 1]  # NOTE: assumes input feature index 1 is "missingness"
        causal_edge_scores_u = causal_edge_scores.unsqueeze(1)  # (B, 1, C, C)
        parents_missing_indicator = causal_edge_scores_u * masking_indicator.unsqueeze(2)  # (B, N, C, C)
        parents_missing_indicator = parents_missing_indicator.sum(dim=3)  # (B, N, C)

        hetero = self.continuous_heteroscedastic_head(current_state)  # (B, N, C, 2)
        pred_mean = hetero[..., 0]  # (B, N, C)

        plb_row = self.continuous_unavailable_logit_bias(current_state)  # (B, N, C, 3)

        valid_row_nc = (~row_mask).view(B, N, 1, 1).float()  # (B,N,1,1)
        plb = (plb_row * valid_row_nc).sum(dim=1) / valid_row_nc.sum(dim=1).clamp_min(1.0)  # (B,C,3)

        base = plb[:, :, 0]  # (B,C)
        raw_some = plb[:, :, 1]  # (B,C)
        raw_none = plb[:, :, 2]  # (B,C)

        # build the log-variance channel out-of-place
        base_var = base.unsqueeze(1).expand_as(pred_mean)  # (B,N,C)
        added_var = parents_missing_indicator * raw_some.unsqueeze(1)  # (B,N,C)
        pred_log_var = base_var + added_var

        heteroscedastic_output = torch.stack([pred_mean, pred_log_var], dim=-1)  # (B,N,C,2)

        addon_some = nn.functional.softplus(raw_some)
        addon_none = nn.functional.softplus(raw_none)

        logit_all = base
        logit_some = addon_some
        logit_none = addon_some + addon_none
        parent_logit_bias = torch.stack([logit_all, logit_some, logit_none], dim=-1)  # (B,C,3)

        # OPTIONAL (recommended): if context is not part of the causal graph, force edges involving it to 0
        if self.context_col_idx is not None:
            k = int(self.context_col_idx)

            M = torch.ones((C, C), device=causal_edge_scores.device, dtype=causal_edge_scores.dtype)
            M[k, :] = 0.0
            M[:, k] = 0.0
            M = M.unsqueeze(0)  # (1,C,C)

            causal_edge_scores = causal_edge_scores * M
            parent_weights = parent_weights * M
            p_ancestor = p_ancestor * M

        return (
            causal_edge_scores,
            p_ancestor,
            learned_order_scores,
            parents_missing_indicator,
            heteroscedastic_output,
            parent_logit_bias,
            parent_weights,
        )
