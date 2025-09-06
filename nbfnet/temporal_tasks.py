# temporal_tasks.py
import math
import torch
from torch.utils import data as torch_data
import torch.nn.functional as F

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.TemporalKnowledgeGraphCompletion")
class TemporalKnowledgeGraphCompletion(tasks.Task, core.Configurable):
    """
    A temporal replacement for your KnowledgeGraphCompletionExt that
    expects batches of (h, t, r, time). It:
      - uses fact_graph (with edge_time) for negatives / filtering,
      - passes query_time to the temporal model,
      - computes filtered ranking with facts up to the query time.
    """

    _option_members = ["criterion", "metric"]

    def __init__(self, model, criterion="bce",
                 metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
                 num_negative=100, strict_negative=True, filtered_ranking=True,
                 full_batch_eval=True,
                 debug=False):
        super().__init__()
        self.model = model
        self.criterion = {"bce": 1.0} if isinstance(criterion, str) else criterion
        self.metric = metric
        self.num_negative = num_negative
        self.strict_negative = strict_negative
        self.filtered_ranking = filtered_ranking
        self.full_batch_eval = full_batch_eval
        self.debug = debug
        self.progress = 0

    def preprocess(self, train_set, valid_set, test_set):
        if self.debug:
            print("[DEBUG] Entering preprocess in TemporalKnowledgeGraphCompletion")
        dataset = train_set.dataset if isinstance(train_set, torch_data.Subset) else train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("fact_graph", dataset.graph)  # carries edge_time
        if self.debug:
            print(f"[DEBUG] Using graph with {self.fact_graph.num_node} nodes and {self.fact_graph.num_edge} edges.")
            print("[DEBUG] Exiting preprocess")
        return train_set, valid_set, test_set

    @torch.no_grad()
    def _strict_negative(self, pos_h, pos_t, pos_r, pos_time):
        """
        Time-aware negatives that avoid any fact (h, t', r) or (h', t, r)
        that existed BEFORE OR AT pos_time. We sample uniformly from entities
        that are not observed by that time.
        """
        B = len(pos_h)
        any = -torch.ones_like(pos_h)
        # Build masks for t candidates
        pattern_t = torch.stack([pos_h, any, pos_r], dim=-1)  # [B, 3]
        edge_index, num_t_truth = self.fact_graph.match(pattern_t)
        t_truth = self.fact_graph.edge_list[edge_index, 1]
        t_time = self.fact_graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_t_truth)
        # valid truths before or at query time
        valid_truth = t_time <= pos_time[pos_index]
        # candidates mask
        t_mask = torch.ones(B, self.num_entity, dtype=torch.bool, device=self.device)
        t_mask[pos_index[valid_truth], t_truth[valid_truth]] = 0
        t_mask[torch.arange(B, device=self.device), pos_h] = 0  # avoid self-loops if needed

        t_cand = t_mask.nonzero()[:, 1]
        t_num = t_mask.sum(dim=-1)
        neg_t = functional.variadic_sample(t_cand, t_num, self.num_negative)  # [B, K]

        # Similarly for head
        pattern_h = torch.stack([any, pos_t, pos_r], dim=-1)
        edge_index, num_h_truth = self.fact_graph.match(pattern_h)
        h_truth = self.fact_graph.edge_list[edge_index, 0]
        h_time = self.fact_graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_h_truth)
        valid_truth = h_time <= pos_time[pos_index]
        h_mask = torch.ones(B, self.num_entity, dtype=torch.bool, device=self.device)
        h_mask[pos_index[valid_truth], h_truth[valid_truth]] = 0
        h_mask[torch.arange(B, device=self.device), pos_t] = 0

        h_cand = h_mask.nonzero()[:, 1]
        h_num = h_mask.sum(dim=-1)
        neg_h = functional.variadic_sample(h_cand, h_num, self.num_negative)  # [B, K]

        # Concatenate tail-negs and head-negs split half/half to mimic your original scheme
        neg_index = torch.cat([neg_t, neg_h], dim=0)  # [2B, K]
        return neg_index  # used in training

    def predict(self, batch, all_loss=None, metric=None):
        if self.debug:
            if isinstance(batch, torch.Tensor):
                print(f"[DEBUG] Entering predict. Batch shape: {batch.shape}")
            else:
                print(f"[DEBUG] Entering predict. Batch type: {type(batch)}, length: {len(batch)}")
        """
        batch: LongTensor [B, 4] -> (h, t, r, time)
        returns logits: [B, K+1]
        """
        if isinstance(batch, list):
            batch = torch.stack(batch, dim=0)
        pos_h, pos_t, pos_r, pos_time = batch.t()
        B = len(batch)
        graph = self.fact_graph
        if self.debug:
            print(f"[DEBUG] pos_h shape: {pos_h.shape}, pos_t shape: {pos_t.shape}, pos_r shape: {pos_r.shape}, pos_time shape: {pos_time.shape}")

        if all_loss is None:
            if self.debug:
                print("[DEBUG] Evaluation mode in predict")
            # eval: score against all entities (or chunks) at the same time
            all_index = torch.arange(graph.num_node, device=self.device)
            t_preds = []
            h_preds = []
            print(all_index)
            num_negative = graph.num_node if self.full_batch_eval else self.num_negative
            # print(f"[DEBUG] num_negative: {num_negative}")
            for neg_chunk in all_index.split(num_negative):
                r_index = pos_r.unsqueeze(-1).expand(-1, len(neg_chunk))
                q_time = pos_time
                # tail ranking
                # print(f"negative {neg_chunk} pos_h {pos_h} pos_t {pos_t}")
                h_index, t_index = torch.meshgrid(pos_h, neg_chunk)
                # print(f"[DEBUG] h_index shape: {h_index.shape} {h_index}, t_index shape: {t_index.shape} {t_index}")
                t_pred = self.model(graph, h_index, t_index, r_index, query_time=q_time,
                                    all_loss=all_loss, metric=metric)
                # print(f"[DEBUG] t_pred shape: {t_pred.shape} {t_pred}")
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            # print(f"t+pred affter {t_pred.shape} {t_pred}")

            for neg_chunk in all_index.split(num_negative):
                r_index = pos_r.unsqueeze(-1).expand(-1, len(neg_chunk))
                q_time = pos_time
                # head ranking
                t_index, h_index = torch.meshgrid(pos_t, neg_chunk)
                # print(f"t_index{t_index} h_index 143 {h_index}")
                h_pred = self.model(graph, h_index, t_index, r_index, query_time=q_time,
                                    all_loss=all_loss, metric=metric)
                # print(f"h_pred shape: {h_pred.shape} {h_pred}")
                h_preds.append(h_pred)
            h_pred = torch.cat(h_preds, dim=-1)

            pred = torch.stack([t_pred, h_pred], dim=1).cpu()
            # print(f"final affter {pred.shape} {pred}")

        else:
            if self.debug:
                print("[DEBUG] Training mode in predict")
            # train: compose negatives at same time as positive
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h, pos_t, pos_r, pos_time)  # [2B, K]
            else:
                neg_index = torch.randint(self.num_entity, (2 * B, self.num_negative))

            h_index = pos_h.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r.unsqueeze(-1).repeat(1, self.num_negative + 1)
            q_time  = pos_time

            # half tail-negatives, half head-negatives
            # Prepare r_index and q_time for both predictions
            r_index_tail = pos_r.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index_head = pos_r.unsqueeze(-1).repeat(1, self.num_negative + 1)
            q_time_tail = pos_time
            q_time_head = pos_time

            # Tail prediction: h_index fixed, t_index varies
            h_index_tail = pos_h.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index_tail = torch.cat([pos_t.unsqueeze(-1), neg_index[:B]], dim=1)
            pred_tail = self.model(graph, h_index_tail, t_index_tail, r_index_tail, query_time=q_time_tail,
                                    all_loss=all_loss, metric=metric)

            # Head prediction: t_index fixed, h_index varies
            t_index_head = pos_t.unsqueeze(-1).repeat(1, self.num_negative + 1)
            h_index_head = torch.cat([pos_h.unsqueeze(-1), neg_index[B:]], dim=1)
            pred_head = self.model(graph, h_index_head, t_index_head, r_index_head, query_time=q_time_head,
                                    all_loss=all_loss, metric=metric)
            

            # print(f"pred_tail : {pred_tail.shape} {pred_tail}")
            # print(f"pred_head : {pred_head.shape} {pred_head}")
            # Stack predictions for compatibility with evaluation
            pred = torch.stack([pred_tail, pred_head], dim=1)
            if self.debug:
                print(f"[DEBUG] pred shape: {pred.shape}")

            # pred = self.model(graph, h_index, t_index, r_index, query_time=q_time,
            #                   all_loss=all_loss, metric=metric)
        if self.debug:
            print("[DEBUG] Exiting predict")
        return pred

    def target(self, batch):
        # if self.debug:
        # print(f"[DEBUG] Entering target. Batch shape: {batch.shape}")
        # evaluation: produce time-filtered masks for rankings
        B = len(batch)
        graph = self.fact_graph
        pos_h, pos_t, pos_r, pos_time = batch.t()
        any = -torch.ones_like(pos_h)

        # Tail mask
        pattern = torch.stack([pos_h, any, pos_r], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth = graph.edge_list[edge_index, 1]
        t_time = graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_t_truth)
        valid_truth = t_time <= pos_time[pos_index]
        t_mask = torch.ones(B, graph.num_node, dtype=torch.bool, device=self.device)
        t_mask[pos_index[valid_truth], t_truth[valid_truth]] = 0

        # Head mask
        pattern = torch.stack([any, pos_t, pos_r], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth = graph.edge_list[edge_index, 0]
        h_time = graph.edge_time[edge_index]
        pos_index = torch.repeat_interleave(num_h_truth)
        valid_truth = h_time <= pos_time[pos_index]
        h_mask = torch.ones(B, graph.num_node, dtype=torch.bool, device=self.device)
        h_mask[pos_index[valid_truth], h_truth[valid_truth]] = 0

        # For negative sampling, mask and target should be aligned with candidate indices
        # mask = torch.ones((B, 2, self.num_negative + 1), dtype=torch.bool).cpu()
        # target = torch.zeros((B, 2), dtype=torch.long).cpu()
        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t, pos_h], dim=1)
        if self.debug:
            print(f"[DEBUG] target shape: {target}")
            print(f"[DEBUG] mask shape: {mask.shape}")
            print("[DEBUG] Exiting target")
        print(mask[0, 0, target[0, 0]])  # tail mask at true target
        print(mask[0, 1, target[0, 1]])  # head mask at true target

        
        return mask, target
    
    def evaluate(self, pred, target):
      
        self.progress += 1

        mask, target = target
        print(f"Mask : {mask.shape} {mask}")
        print(f"Target : {target.shape} {target}")
        if self.progress % 100 == 0:
            print(f"[DEBUG] Entering evaluate. Pred shape: {pred.shape}, Target shape: {target.shape}")
        print(f"Pred : {pred.shape}")
        print("Gold in mask (tail):", mask[0,0,target[0,0]])
        print("Gold in mask (head):", mask[0,1,target[0,1]])

        pos_pred = pred.gather(-1, target.to(pred.device).unsqueeze(-1))
        print(f"Pos_pred : {pos_pred.shape} {pos_pred}")
        ranking = torch.sum((pos_pred <= pred) & mask.to(pred.device), dim=-1) + 1
        print(f"Ranking : {ranking.shape} {ranking}")
        metric = {}
        ranking = torch.minimum(ranking[:, 0],pred.shape[2]-ranking[:, 1])
        print(f"Ranking : {ranking.shape} {ranking}")

        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                values = _metric[5:].split("_")
                threshold = int(values[0])
                if len(values) > 1:
                    num_sample = int(values[1])
                    # unbiased estimation
                    fp_rate = (ranking - 1).float() / mask.sum(dim=-1)
                    score = 0
                    for i in range(threshold):
                        # choose i false positive from num_sample negatives
                        num_comb = math.factorial(num_sample) / math.factorial(i) / math.factorial(num_sample - i)
                        score += num_comb * (fp_rate ** i) * ((1 - fp_rate) ** (num_sample - i))
                    score = score.mean()
                else:
                    score = (ranking <= threshold).float().mean()
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


    def forward(self, batch):
        if self.debug:
            print(f"[DEBUG] Entering forward. Batch : {batch}")
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        # pred = self.predict(batch, all_loss, metric)
        # if self.debug:
        #     print(f"[DEBUG] pred shape after predict: {pred.shape}")
        # mask, target = self.target(batch)  # on CPU to avoid OOM
        # if self.debug:
        #     print(f"[DEBUG] mask shape: {mask.shape}, target shape: {target.shape}")

        #     # compute filtered ranking metrics
        # pos_pred = pred.gather(-1, target.to(self.device).unsqueeze(-1))
        # if self.debug:
        #     print(f"[DEBUG] pos_pred shape: {pos_pred.shape}")
        # Broadcast pos_pred to match pred and mask shapes
        # pos_pred_expanded = pos_pred.expand_as(pred)
        # ranking = torch.sum((pos_pred_expanded <= pred) & mask.to(self.device), dim=-1) + 1  # [B, 2]
        # if self.debug:
        #     print(f"[DEBUG] Ranking: {ranking}")
        out = {}
        # for _m in self.metric:
        #     if _m == "mr":
        #         score = ranking.float().mean()
        #     elif _m == "mrr":
        #         score = (1 / ranking.float()).mean()
        #     elif _m.startswith("hits@"):
        #         k = int(_m.split("@")[1])
        #         score = (ranking <= k).float().mean()
        #     else:
        #         raise ValueError(f"Unknown metric `{_m}`")
        #     out[tasks._get_metric_name(_m)] = score

        # loss (binary cross-entropy on packed positives + negatives)
        loss = torch.tensor(0.0, device=self.device)
        if "bce" in self.criterion:
            # Construct logits/labels like your LinkPrediction: one positive + K negatives
            # We already have pred for train path inside predict(); re-compute simply here for loss
            pos = batch[:, :3]  # (h, t, r)
            # re-use training path to get per-sample (1 + K) logits
            tr_logits = self.predict(batch, all_loss=torch.tensor(0.0, device=self.device))
            labels = torch.zeros_like(tr_logits)
            labels[:, 0] = 1.0
            neg_weight = torch.ones_like(tr_logits)
            if tr_logits.size(1) > 1:
                neg_weight[:, 1:] = 1.0 / (tr_logits.size(1) - 1)
            bce = F.binary_cross_entropy_with_logits(tr_logits, labels, reduction="none")
            bce = (bce * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = bce.mean()

        out[tasks._get_criterion_name("bce")] = loss
        all_loss = loss
        if self.debug:
            print(f"[DEBUG] Loss: {all_loss.item()}, Metrics: {out}")
            print("[DEBUG] Exiting forward")

        return all_loss, out
