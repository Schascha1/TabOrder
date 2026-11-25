"""TabOrder architecture, preserving the research checkpoint parameter names."""

from typing import Optional
import time

import torch
from torch import nn

class SetAttentionBlock(nn.Module):
    def __init__(self, d_model, num_layers=1, num_heads=8, d_ff=None, dropout=0.1, prenorm=True):
        super().__init__()
        self.num_heads = num_heads  # expose for building (BN*H, L, L) masks
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
        # attn_mask can be:
        #  - (L, L) shared across batch, OR
        #  - (B*num_heads, L, L) for per-sample masks (what we need)
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
            batch_first=True,   # so we can pass (B, C, D)
        )
        self.forbid_self = forbid_self

    def forward(self, h: torch.Tensor, col_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h:        (B, C, D)  per-variable embeddings
        col_mask: (B, C)     True = padded column
        Returns:
            parent_weights: (B, C, C)  parent_weights[b, i, j] ~ weight j -> i
        """
        B, C, D = h.shape

        if self.forbid_self:
            if attn_mask is None:
            # (C, C) additive mask shared across batch: diag = -inf
                attn_mask = torch.zeros(C, C, device=h.device)
                attn_mask.fill_(0.0)
                attn_mask.fill_diagonal_(float("-inf"))
            else:
                attn_mask[:,torch.eye(C, device=h.device, dtype=torch.bool)] = float("-inf")
        

        # MultiheadAttention returns:
        # - attn_output: (B, C, D)
        # - attn_weights: (B, C, C)
        attn_output, attn_weights = self.attn(
            query=h,
            key=h,
            value=h,
            key_padding_mask=col_mask,  # True = ignore that variable as key/value
            attn_mask=attn_mask,        # optional self-forbid
            need_weights=True,
            #average_attn_weights=True,  # (B, C, C)
        )

        # attn_weights[b, i, j] is how much variable i attends to variable j
        parent_weights = attn_weights  # (B, C, C), j -> i
        #print(parent_weights[0])
        # make into 0-1 range, i.e. probabilities of parentage
        return parent_weights


class CausalOrderModel(nn.Module):
    """
    a Causal Order Model using Set Attention Blocks for column and row attention
    """
    def __init__(self, d_model,d_classifier, d_embedding=128, K_attention_blocks_order=2, K_attention_blocks_pred=3, heads_col_sab_order=4, heads_row_sab_order=4, num_layers_col_sab_order=1, num_layers_row_sab_order=1,
                 heads_col_sab_pred=4, heads_row_sab_pred=4, num_layers_col_sab_pred=1, num_layers_row_sab_pred=1, temperature_order=1.0, attention_bias=-5, fix_inplace=False, order_mode="causal", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_inplace = fix_inplace
        self.order_mode = order_mode
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
        self.edge_mlp = nn.Sequential(
            nn.Linear(4*d_embedding, 2*d_embedding),
            nn.ReLU(),
            nn.Linear(2*d_embedding, 1)
        )
        self.parent_head = ParentAttentionHead(d_embedding, forbid_self=True)
        self.temperature_order = temperature_order
        self.attention_bias = attention_bias


    def forward(self, x, col_mask=None, row_mask=None, query_indicator=None):
        """
        x: (B, N, C, D_in), channels = value, missing flag, type, categories
        col_mask: (B, C)    # bool, True=PAD column
        row_mask: (B, N) # bool, True=PAD row
        Returns:
          edge_scores, pairwise_order, order_scores, missing_parent_scores,
          mean_and_log_variance, variance_bias_terms, parent_attention
        """
        t = time.time()
        B, N, C, D = x.size()
        # masking out padded columns and rows
        repeated_col_mask = col_mask.repeat_interleave(N,dim=0)
        row_mask_bn = row_mask.repeat_interleave(C,dim=0)

        embedded_input  = self.cell_embedder(x[:, :,:,:])          # (B, N, C, d_embedding)

        current_state = embedded_input

        # process order prediction
        for i in range(len(self.order_col_attention_blocks)):

    
            col_attention_order = self.order_col_attention_blocks[i]
            row_attention_order = self.order_row_attention_blocks[i]

            # batch per row
            row_batches = current_state.reshape(B * N, C, self.d_embedding)
            col_embeddings = col_attention_order(row_batches, key_padding_mask=repeated_col_mask)
            col_embeddings = col_embeddings.reshape(B, N, C, self.d_embedding).transpose(1,2)
            # batch per column
            col_embeddings = col_embeddings.reshape(B * C, N, self.d_embedding)
            col_embeddings = row_attention_order(col_embeddings, key_padding_mask=row_mask_bn)
            current_state = col_embeddings.reshape(B, C, N, self.d_embedding).transpose(1,2)

        # create order score per batch, per column
        order_scores = self.order_decoder(current_state).transpose(1,2) # (B, C, N, 1)
        # aggregate by mean 
        valid_row = (~row_mask_bn).view(B,C,N,1).float()
        order_scores = order_scores * valid_row

        order_scores = order_scores.sum(dim=2) / torch.clamp_min(valid_row.sum(dim=2), 1)  # (B, C, 1)
        order_scores = order_scores.squeeze(-1)  # (B, C)

        order_mean = order_scores.mean(dim=1, keepdim=True)
        order_std  = order_scores.std(dim=1, keepdim=True).clamp_min(1e-3)
        order_scores = (order_scores - order_mean) / order_std

        if self.order_mode == "noorder":
            p_ancestor = torch.ones((B, C, C), device=x.device, dtype=x.dtype)
            eye = torch.eye(C, device=x.device, dtype=torch.bool)[None]
            p_ancestor = p_ancestor.masked_fill(eye, 0.0)
            attn_mask = None
        else:
            # delta_ij = score_i - score_j
            # if score_i > score_j => i is an ancestor of j => i may attend to j
            delta = (order_scores[:,:,None] - order_scores[:,:,None].transpose(1,2))/self.temperature_order  # (B, C, C)

            # p_ancestor[i,j] ~ probability that j is ancestor of i
            p_ancestor = torch.sigmoid(delta * 1)
            if not self.training:
                p_ancestor = (p_ancestor >= 0.5).float()

            #soft_mask = self.attention_bias * (1.0 - p_ancestor)  # (B, C, C)

            #hard_allow = (p_ancestor >= 0.5)
            #hard_mask = torch.where(
            #    hard_allow,
            #    torch.zeros_like(delta),
            #    self.attention_bias * torch.ones_like(delta),
            #)

            #eye = torch.eye(C, device=delta.device, dtype=torch.bool)[None]  # (1, C, C)
            #soft_mask = torch.where(eye, torch.zeros_like(soft_mask), soft_mask)
            #hard_mask = torch.where(eye, torch.zeros_like(hard_mask), hard_mask)

            # straight-through estimator:
            #attn_mask = hard_mask.detach() + (soft_mask - soft_mask.detach())

            eye = torch.eye(
                C,
                device=delta.device,
                dtype=torch.bool,
            )[None]

            # A variable may attend to its causal predecessors and to itself.
            hard_allow = (p_ancestor >= 0.5) | eye

            if not self.training:
                # Strictly exclude later variables during evaluation.
                attn_mask = torch.where(
                    hard_allow,
                    torch.zeros_like(delta),
                    torch.full_like(
                        delta,
                        torch.finfo(delta.dtype).min,
                    ),
                )
            else:
                # Retain the original finite straight-through mask during training.
                soft_mask = self.attention_bias * (1.0 - p_ancestor)

                hard_mask = torch.where(
                    hard_allow,
                    torch.zeros_like(delta),
                    self.attention_bias * torch.ones_like(delta),
                )

                soft_mask = torch.where(
                    eye,
                    torch.zeros_like(soft_mask),
                    soft_mask,
                )

                attn_mask = (
                        hard_mask.detach()
                        + (soft_mask - soft_mask.detach())
                )

        # predict causal parentage
        # valid mask
        valid_row_causal = (~row_mask).view(B, N, 1, 1).float()     # (B, N, 1, 1)
        valid_col = (~col_mask).view(B, 1, C, 1).float()     # (B, 1, C, 1)
        valid_row_causal = valid_row_causal * valid_col                        # (B, N, C, 1)

        denom = valid_row_causal.sum(dim=1).clamp_min(1.0)              # (B, C, 1)
        h = (current_state * valid_row_causal).sum(dim=1) / denom       # (B, C, D)

        # child = h[:, :, None, :].repeat(1, 1, C, 1)          # (B, C, 1, D)
        # parent = h[:, None, :, :].repeat(1, C, 1, 1)         # (B, 1, C, D)

        # edge_feat = torch.cat([child, parent, child*parent, child-parent], dim=-1)  # (B,C,C,4D)
        # parent_weights = self.edge_mlp(edge_feat).squeeze(-1)  # (B,C,C)
        
        # parent_weights = torch.nn.functional.softplus(parent_weights)
        # # zero out diagonal
        
        # log_var_add = torch.softmax(parent_weights, dim=-1)  # (B, C)

        # log_var_add = parent_weights *p_ancestor
        # log_var_add[eye.repeat(B,1,1)] = 0.0
        # causal_edge_scores = log_var_add

        # h[b, c, :] is the embedding for variable c in batch b
        #parent_weights = self.parent_head(h, col_mask, attn_mask=attn_mask)  # (B, C, C)

        parent_attn_mask = (
            None if attn_mask is None else attn_mask.clone()
        )

        parent_weights = self.parent_head(
            h,
            col_mask,
            attn_mask=parent_attn_mask,
        )
        causal_edge_scores = parent_weights*p_ancestor

        # # attention mask for predicting parents
        # # if score_i > score_j => scores_cmp > 0 j may be parent of i => allow attention
        # scores_cmp = torch.nn.functional.logsigmoid(-scores_cmp) 
        
        # additive bias to row attention pred
        if attn_mask is not None:
            attn_bias_bn = attn_mask.repeat_interleave(N, dim=0)   # (B*N, C, C)
            H = self.pred_col_attention_blocks[0].num_heads
            src_mask = attn_bias_bn.repeat_interleave(H, dim=0)    # (BN*H, C, C)
        else:
            src_mask = None

        current_state = embedded_input

        for i in range(len(self.pred_col_attention_blocks)):
    
            col_attention_pred = self.pred_col_attention_blocks[i]
            row_attention_pred = self.pred_row_attention_blocks[i]

            # batch per row
            row_batches = current_state.reshape(B * N, C, self.d_embedding)
            col_embeddings = col_attention_pred(row_batches, key_padding_mask=repeated_col_mask
                                                ,attn_mask=src_mask
                                                )
            col_embeddings = col_embeddings.reshape(B, N, C, self.d_embedding).transpose(1,2)
            # batch per column
            col_embeddings = col_embeddings.reshape(B * C, N, self.d_embedding)
            col_embeddings = row_attention_pred(col_embeddings, key_padding_mask=row_mask_bn)
            current_state = col_embeddings.reshape(B, C, N, self.d_embedding).transpose(1,2)
        

        #continuous_output = self.continous_cell_decoder(current_state).squeeze(-1)
        #continuous_output = continuous_output
        # predicted_parent_availability = self.parent_availability_head(current_state)
        masking_indicator = x[:,:,:,1]
        causal_edge_scores = causal_edge_scores.unsqueeze(1)
        parents_missing_indicator = causal_edge_scores*masking_indicator.unsqueeze(2) # (B, N, C, C)
        parents_missing_indicator = parents_missing_indicator.sum(dim=3) # compute whether any parent is missing
        causal_edge_scores = causal_edge_scores.squeeze(1)

        heteroscedastic_output = self.continuous_heteroscedastic_head(current_state)  # (B, N, C, 2)

        parent_logit_bias = self.continuous_unavailable_logit_bias(current_state)  # (B, N, C, 3)
        valid_row = valid_row.transpose(1,2)
        parent_logit_bias = (parent_logit_bias*valid_row).sum(dim=1) / valid_row.sum(dim=1)

        base = parent_logit_bias[:, :, 0]      # (B, C)
        added_var_through_missing_parents = parents_missing_indicator*parent_logit_bias[:,:,1].unsqueeze(1)

        if self.fix_inplace:
            # out-of-place build: clean autograd graph for both pred_mean and pred_log_var
            pred_mean = heteroscedastic_output[:, :, :, 0]
            pred_log_var = base.unsqueeze(1) + added_var_through_missing_parents
            heteroscedastic_output = torch.stack([pred_mean, pred_log_var], dim=-1)
        else:
            # original in-place behaviour (preserved for backward compatibility)
            heteroscedastic_output[:,:, :,1] = base.unsqueeze(1)
            heteroscedastic_output[:,:, :,1] = heteroscedastic_output[:,:, :,1] + added_var_through_missing_parents

        # raw addons from channels 1 and 2
        raw_some = parent_logit_bias[:, :, 1]   # (B, C)
        raw_none = parent_logit_bias[:, :, 2]   # (B, C)

        # make them positive addons
        addon_some = nn.functional.softplus(raw_some)          # (B, C)
        addon_none = nn.functional.softplus(raw_none)          # (B, C)

        # enforce ordering:
        #   some > all
        #   none > some
        logit_all  = base                                   # (B, C)
        logit_some = addon_some                     # (B, C)
        logit_none = addon_some + addon_none         # (B, C)

        # reassemble into a new tensor, no in-place mutation
        parent_logit_bias = torch.stack(
            [logit_all, logit_some, logit_none], dim=-1
        )  # (B, C, 3)  


        return causal_edge_scores, p_ancestor, order_scores, parents_missing_indicator, heteroscedastic_output, parent_logit_bias, parent_weights


def topological_divergence(node_order_scores, dag):
    # for each node gets its order index
    order = torch.argsort(node_order_scores)
    node_to_order_idx = {node.item(): idx for idx, node in enumerate(order)}
    # dag is networkx graph
    div_top = 0
    for cause, effect in dag.edges():
        cause_idx = node_to_order_idx[cause]
        effect_idx = node_to_order_idx[effect]
        if cause_idx > effect_idx:
            div_top += 1
    return div_top / max(dag.number_of_edges(),1)
