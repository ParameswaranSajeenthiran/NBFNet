"""
Temporal extensions for NBFNet & TorchDrug tasks
------------------------------------------------
This file modifies the provided NBFNet, GeneralizedRelationalConv and Tasks to handle temporal graphs.
Key additions:
- Edge timestamps (graph.edge_time: LongTensor[num_edge])
- Per-query timestamps (query_time: Float/Long tensor [batch])
- Time masking: only edges with time <= query_time (and optional sliding window) contribute
- Optional time encoding and decay in messages
- Temporal-aware negative sampling & filtered ranking

NOTE: For simplicity & correctness, temporal mode disables the fast path in
`message_and_aggregate` and uses the explicit message() + aggregate() route.
"""
from collections.abc import Sequence
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import core, layers, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R

# -----------------------------------------------------------------------------
# Utility: sinusoidal time encoding (can be swapped for learned encodings)
# -----------------------------------------------------------------------------
class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int, max_period: float = 1e4):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        assert dim % 2 == 0, "Time encoding dim should be even"

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: shape [...], returns [..., dim]
        Values are mapped with sinusoidal positional encodings.
        """
        half = self.dim // 2
        device = t.device
        # normalize to [0,1] roughly if large timestamps are used
        # you can replace this with dataset-specific scaling
        scales = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(self.max_period, device=device)) / (half - 1))
        )  # [half]
        ang = t.float().unsqueeze(-1) * scales  # [..., half]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

# -----------------------------------------------------------------------------
# GeneralizedRelationalConv with temporal awareness
# -----------------------------------------------------------------------------
class GeneralizedRelationalConvTemporal(layers.MessagePassingBase):
    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_relation: int,
        query_input_dim: int,
        message_func: str = "distmult",
        aggregate_func: str = "pna",
        layer_norm: bool = False,
        activation: str | callable = "relu",
        dependent: bool = True,
        time_encoding_dim: int = 32,
        time_decay: Optional[str] = None,  # None | 'exp' | 'linear'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent
        self.time_decay = time_decay

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        self.activation = getattr(F, activation) if isinstance(activation, str) else activation

        # feature shaping – we append boundary and (optionally) time encodings into the message
        self.time_encoder = SinusoidalTimeEncoding(time_encoding_dim) if time_encoding_dim > 0 else None
        # base dim from node->edge message
        base_msg_dim = input_dim
        extra_msg_dim = 0
        if self.time_encoder is not None:
            extra_msg_dim += time_encoding_dim
        # boundary is concatenated later by caller (kept consistent with original),
        # so the concatenation here is: [edge_message, time_encoding]

        if self.aggregate_func == "pna":
            self.linear = nn.Linear((base_msg_dim + extra_msg_dim) * 13, output_dim)
        else:
            self.linear = nn.Linear((base_msg_dim + extra_msg_dim) * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

    def _edge_temporal_mask_and_delta(self, graph, batch_size: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns:
        - edge_mask: [num_edge, batch] bool mask, True if edge is usable for that query
        - delta_t:   [num_edge, batch] time difference (query_time - edge_time) or None
        Expects:
        - graph.edge_time: [num_edge] tensor of timestamps
        - graph.query_time: [batch] tensor of timestamps per query
        Optional window: graph.time_window (scalar or None). If set, only keep edges with delta_t in [0, window].
        """
        assert hasattr(graph, "edge_time"), "Temporal graphs must set graph.edge_time"
        assert hasattr(graph, "query_time"), "graph.query_time must be set per batch"
        et = graph.edge_time.view(-1, 1)  # [E, 1]
        qt = graph.query_time.view(1, -1)  # [1, B]
        delta = qt - et  # [E, B]
        edge_mask = delta >= 0  # only past edges
        if hasattr(graph, "time_window") and graph.time_window is not None:
            edge_mask = edge_mask & (delta <= graph.time_window)
        return edge_mask, delta

    def _apply_time_decay(self, delta: torch.Tensor) -> torch.Tensor:
        """delta: [E, B] non-negative floats/ints
        returns a multiplicative weight [E, B]
        """
        if self.time_decay is None:
            return torch.ones_like(delta, dtype=torch.float32)
        d = delta.float()
        if self.time_decay == "exp":
            return torch.exp(-d)
        if self.time_decay == "linear":
            return 1.0 / (1.0 + d)
        raise ValueError(f"Unknown time_decay: {self.time_decay}")

    def message(self, graph, input):
        # Temporal mode uses explicit message() (fast path is disabled below)
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        relation_input = relation_input.transpose(0, 1)  # [R, B, D]
        node_input = input[node_in]  # [E, B, D]
        edge_input = relation_input[relation]  # [E, B, D]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError(f"Unknown message function `{self.message_func}`")

        # Temporal mask & weights
        edge_mask, delta = self._edge_temporal_mask_and_delta(graph, batch_size)
        edge_weight_time = self._apply_time_decay(delta).unsqueeze(-1)  # [E, B, 1]
        message = message * edge_weight_time
        message = message * edge_mask.unsqueeze(-1)  # zero-out future edges

        # Optional time encoding concatenation
        if self.time_encoder is not None:
            te = self.time_encoder(delta.clamp_min(0))  # [E, B, T]
            message = torch.cat([message, te], dim=-1)

        # Append boundary like original layer does (boundary already [N, B, D]; will be added in aggregate)
        # Here we keep only the edge message; boundary is added in aggregate() stage like the original impl.
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        # Append self-loop (boundary) like original implementation does
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)

        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            update = update + graph.boundary
        elif self.aggregate_func == "mean":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            update = (update + graph.boundary) / degree_out
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            update = torch.max(update, graph.boundary)
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            mmax = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            mmin = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), mmax.unsqueeze(-1), mmin.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / (scale.mean() + 1e-9)
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError(f"Unknown aggregation function `{self.aggregate_func}`")

        return update

    def message_and_aggregate(self, graph, input):
        # Disable fast path in temporal mode (due to per-edge, per-batch masks)
        return super().message_and_aggregate(graph, input)

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

# -----------------------------------------------------------------------------
# Temporal NBFNet
# -----------------------------------------------------------------------------
@R.register("model.NBFNetTemporal")
class NeuralBellmanFordNetworkTemporal(nn.Module, core.Configurable):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation=None,
        symmetric=False,
        message_func="distmult",
        aggregate_func="pna",
        short_cut=False,
        layer_norm=False,
        activation="relu",
        concat_hidden=False,
        num_mlp_layer=2,
        dependent=True,
        remove_one_hop=False,
        num_beam=10,
        path_topk=10,
        time_encoding_dim=32,
        time_decay: Optional[str] = "exp",
        time_window: Optional[float] = None,
    ):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation * 2
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk
        self.time_window = time_window

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                GeneralizedRelationalConvTemporal(
                    self.dims[i],
                    self.dims[i + 1],
                    double_relation,
                    self.dims[0],
                    message_func,
                    aggregate_func,
                    layer_norm,
                    activation,
                    dependent,
                    time_encoding_dim=time_encoding_dim,
                    time_decay=time_decay,
                )
            )

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(double_relation, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None, query_time=None):
        # Optionally restrict by time (edges at the same timestamp)
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        # Time gating: keep edges whose time <= query_time (if provided)
        if query_time is not None and hasattr(graph, "edge_time"):
            edge_mask = edge_mask & (graph.edge_time <= query_time)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=True):
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        if self_loop:
            node_in = node_out = torch.arange(graph.num_node, device=self.device)
            loop = torch.stack([node_in, node_out], dim=-1)
            edge_list = torch.cat([edge_list, loop])
            edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
            if hasattr(graph, "edge_time"):
                max_time = graph.edge_time.max() if graph.edge_time.numel() > 0 else torch.tensor(0, device=self.device)
                loop_time = torch.full((graph.num_node,), max_time, device=self.device)
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=self.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        data_kwargs = {**graph.data_dict}
        if hasattr(graph, "edge_time"):
            # append loop_time if self-loops were added
            if self_loop:
                data_kwargs["edge_time"] = torch.cat([graph.edge_time, loop_time])
            else:
                data_kwargs["edge_time"] = graph.edge_time
        graph = type(graph)(
            edge_list,
            edge_weight=edge_weight,
            num_node=graph.num_node,
            num_relation=1,
            meta_dict=graph.meta_dict,
            **data_kwargs,
        )
        return graph

    def bellmanford(self, graph, h_index, r_index, query_time, separate_grad=False):
        self.query = nn.Embedding(graph.num_edge, input_dim)
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(graph.num_node, *query.shape, device=self.device)
        boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        with graph.graph():
            graph.query = query
            graph.query_time = query_time  # [batch]
            graph.time_window = self.time_window
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        step_graphs = []
        layer_input = boundary

        for layer in self.layers:
            if separate_grad:
                step_graph = graph.clone().requires_grad_()
            else:
                step_graph = graph
            hidden = layer(step_graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            step_graphs.append(step_graph)
            layer_input = hidden

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
        }

    def forward(self, graph, h_index, t_index, r_index=None, query_time: Optional[torch.Tensor] = None, all_loss=None, metric=None):
        if query_time is None:
            raise ValueError("Temporal model requires query_time tensor of shape [batch]")

        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index, query_time.max())

        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0], query_time)
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.bellmanford(graph, t_index[:, 0], r_index[:, 0], query_time)
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

# -----------------------------------------------------------------------------
# Temporal Tasks
# -----------------------------------------------------------------------------
@R.register("tasks.TemporalKnowledgeGraphCompletion")
class TemporalKnowledgeGraphCompletion(tasks.KnowledgeGraphCompletion, core.Configurable):
    """Evaluate p(t | h, r, time) and p(h | t, r, time) under temporal constraints."""

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
        num_negative=128,
        margin=6,
        adversarial_temperature=0,
        strict_negative=True,
        sample_weight=True,
        filtered_ranking=True,
        full_batch_eval=False,
    ):
        super().__init__(
            model,
            criterion,
            metric,
            num_negative,
            margin,
            adversarial_temperature,
            strict_negative,
            fact_ratio=None,
            sample_weight=sample_weight,
            filtered_ranking=filtered_ranking,
            full_batch_eval=full_batch_eval,
        )

    def preprocess(self, train_set, valid_set, test_set):
        # Expect datasets to include timestamps as the 4th column in each triple: (h, t, r, time)
        if isinstance(train_set, torch.utils.data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation

        # Graph already should carry edge_time aligned with edge_list
        self.register_buffer("train_graph", dataset.train_graph)
        self.register_buffer("valid_graph", dataset.valid_graph)
        self.register_buffer("test_graph", dataset.test_graph)

        return train_set, valid_set, test_set

    @torch.no_grad()
    def _strict_negative_temporal(self, graph, pos_h, pos_t, pos_r, q_time, num_negative: int):
        """Sample negatives that do not violate observed facts up to time q_time.
        Returns neg_index: [batch, num_negative]
        For the first half of the batch, corrupt tail; second half corrupt head (like TorchDrug default).
        """
        batch = len(pos_h)
        device = graph.device
        neg_index = torch.empty(batch, num_negative, dtype=torch.long, device=device)

        # Build adjacency per relation filtered by time
        # For efficiency, precompute neighbors per (h,r) and (t,r) up to q_time via masking
        E = graph.edge_list.shape[0]
        edge_h = graph.edge_list[:, 0]
        edge_t = graph.edge_list[:, 1]
        edge_r = graph.edge_list[:, 2]
        etime = graph.edge_time

        for i in range(batch):
            h, t, r, qt = pos_h[i].item(), pos_t[i].item(), pos_r[i].item(), q_time[i].item()
            mask = (edge_r == r) & (etime <= qt)
            # tails seen for (h, r)
            tails = edge_t[mask & (edge_h == h)]
            # heads seen for (r, t)
            heads = edge_h[mask & (edge_t == t)]
            if i < batch // 2:
                # corrupt tail
                ban = set(tails.tolist() + [h])
                candidates = torch.tensor([x for x in range(graph.num_node) if x not in ban], device=device)
                if len(candidates) == 0:
                    candidates = torch.arange(graph.num_node, device=device)
                choice = candidates[torch.randint(len(candidates), (num_negative,), device=device)]
                neg_index[i] = choice
            else:
                # corrupt head
                ban = set(heads.tolist() + [t])
                candidates = torch.tensor([x for x in range(graph.num_node) if x not in ban], device=device)
                if len(candidates) == 0:
                    candidates = torch.arange(graph.num_node, device=device)
                choice = candidates[torch.randint(len(candidates), (num_negative,), device=device)]
                neg_index[i] = choice
        return neg_index

    def predict(self, batch, all_loss=None, metric=None):
        # batch: [B, 4] (h, t, r, time)
        pos_h_index, pos_t_index, pos_r_index, pos_time = batch.t()
        graph = getattr(self, f"{self.split}_graph")

        if all_loss is None:
            # test-time: full negative ranking over all nodes (optionally chunked)
            all_index = torch.arange(graph.num_node, device=self.device)
            t_preds = []
            h_preds = []
            num_negative = graph.num_node if self.full_batch_eval else self.num_negative
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                q_time = pos_time
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index, indexing="ij")
                t_pred = self.model(graph, h_index, t_index, r_index, query_time=q_time, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                q_time = pos_time
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index, indexing="ij")
                h_pred = self.model(graph, h_index, t_index, r_index, query_time=q_time, all_loss=all_loss, metric=metric)
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1).cpu()
        else:
            # train-time negative sampling respecting time
            batch_size = len(batch)
            if self.strict_negative:
                neg_index = self._strict_negative_temporal(graph, pos_h_index, pos_t_index, pos_r_index, pos_time, self.num_negative)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            q_time = pos_time  # [B]
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(graph, h_index, t_index, r_index, query_time=q_time, all_loss=all_loss, metric=metric)

        return pred

    def target(self, batch):
        # Build temporal filtered masks like TorchDrug but only up to q_time
        batch_size = len(batch)
        graph = getattr(self, f"{self.split}_graph")
        pos_h_index, pos_t_index, pos_r_index, pos_time = batch.t()
        any = -torch.ones_like(pos_h_index)

        # For simplicity (and because graph.match doesn't natively support time),
        # we construct masks by scanning edge_list.
        E = graph.edge_list.shape[0]
        edge_h = graph.edge_list[:, 0]
        edge_t = graph.edge_list[:, 1]
        edge_r = graph.edge_list[:, 2]
        etime = graph.edge_time

        t_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        h_mask = torch.ones(batch_size, graph.num_node, dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            h, t, r, qt = pos_h_index[i].item(), pos_t_index[i].item(), pos_r_index[i].item(), pos_time[i].item()
            mask = (edge_r == r) & (etime <= qt)
            t_truth = edge_t[mask & (edge_h == h)]
            h_truth = edge_h[mask & (edge_t == t)]
            t_mask[i, t_truth] = 0
            t_mask[i, h] = 0  # exclude self if needed (like original strict setting)
            h_mask[i, h_truth] = 0
            h_mask[i, t] = 0

        target = torch.stack([pos_t_index, pos_h_index], dim=1)
        # CPU in case of OOM like TorchDrug style
        return torch.stack([t_mask, h_mask], dim=1).cpu(), target.cpu()

    def evaluate(self, pred, target):
        mask, target = target
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1

        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).float().mean()
            else:
                raise ValueError(f"Unknown metric `{_metric}`")
            name = tasks._get_metric_name(_metric)
            metric[name] = score
        return metric

# -----------------------------------------------------------------------------
# Minimal Temporal LinkPrediction (no relations)
# -----------------------------------------------------------------------------
@R.register("tasks.TemporalLinkPrediction")
class TemporalLinkPrediction(tasks.Task, core.Configurable):
    _option_members = ["criterion", "metric"]

    def __init__(self, model, criterion="bce", metric=("auroc", "ap"), num_negative=128, strict_negative=True):
        super().__init__()
        self.model = model
        self.criterion = {criterion: 1.0} if isinstance(criterion, str) else criterion
        self.metric = metric
        self.num_negative = num_negative
        self.strict_negative = strict_negative

    def preprocess(self, train_set, valid_set, test_set):
        dataset = train_set.dataset if isinstance(train_set, torch.utils.data.Subset) else train_set
        self.num_node = dataset.num_node
        self.register_buffer("train_graph", dataset.graph.undirected())
        self.register_buffer("valid_graph", dataset.graph.undirected())
        self.register_buffer("test_graph", dataset.graph.undirected())

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                neg_weight = torch.ones_like(pred)
                neg_weight[:, 1:] = 1 / self.num_negative
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            else:
                raise ValueError(f"Unknown criterion `{criterion}`")
            loss = loss.mean()
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight
        return all_loss, metric

    @torch.no_grad()
    def _strict_negative_temporal(self, graph, h_index, t_index, q_time, count):
        # sample negatives not connected up to q_time
        # returns (neg_h_index, neg_t_index) each [count]
        edge_h = graph.edge_list[:, 0]
        edge_t = graph.edge_list[:, 1]
        etime = graph.edge_time

        device = graph.device
        neg_h = torch.empty(count, dtype=torch.long, device=device)
        neg_t = torch.empty(count, dtype=torch.long, device=device)

        for i in range(count):
            qt = q_time[i % len(q_time)]
            # choose a head with available non-neighbors at qt
            h = torch.randint(self.num_node, (1,), device=device).item()
            mask = (edge_h == h) & (etime <= qt)
            forbidden = set(edge_t[mask].tolist() + [h])
            candidates = torch.tensor([x for x in range(self.num_node) if x not in forbidden], device=device)
            if len(candidates) == 0:
                candidates = torch.arange(self.num_node, device=device)
            t_neg = candidates[torch.randint(len(candidates), (1,), device=device)]
            neg_h[i] = h
            neg_t[i] = t_neg
        return neg_h, neg_t

    def predict_and_target(self, batch, all_loss=None, metric=None):
        # batch: [B, 3] (h, t, time)
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_time = batch.t()

        if self.split == "train":
            num_negative = self.num_negative
        else:
            num_negative = 1
        graph = getattr(self, f"{self.split}_graph")

        if self.strict_negative or self.split != "train":
            neg_h_index, neg_t_index = self._strict_negative_temporal(graph, pos_h_index, pos_t_index, pos_time, batch_size * num_negative)
        else:
            neg_h_index, neg_t_index = torch.randint(self.num_node, (2, batch_size * num_negative), device=self.device)
        neg_h_index = neg_h_index.view(batch_size, num_negative)
        neg_t_index = neg_t_index.view(batch_size, num_negative)

        h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
        t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
        q_time = pos_time  # [B]
        h_index[:, 1:] = neg_h_index
        t_index[:, 1:] = neg_t_index

        pred = self.model(graph, h_index, t_index, r_index=None, query_time=q_time, all_loss=all_loss, metric=metric)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        return pred, target

    def evaluate(self, pred, target):
        pred = pred.flatten()
        target = target.flatten()
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc":
                score = metrics.area_under_roc(pred, target)
            elif _metric == "ap":
                score = metrics.area_under_prc(pred, target)
            else:
                raise ValueError(f"Unknown metric `{_metric}`")
            name = tasks._get_metric_name(_metric)
            metric[name] = score
        return metric
