# icews_loader_and_train_local.py
import os
import torch
from torch.utils.data import Dataset, random_split
# from nbfnet import dataset
from torchdrug import data as td_data, core
from torchdrug import utils as td_utils
from torch.utils.data import TensorDataset

from temporal_nbfnet import NeuralBellmanFordNetworkTemporal
from temporal_tasks import TemporalKnowledgeGraphCompletion


# ---------------------------
# Local Dataset Loader
# ---------------------------

def load_local_icews(data_dir="data/ICEWS14", valid_ratio=0.4, seed=42):
    """
    Load ICEWS14 from local txt files and split train into train/valid.
    Expected format per line: head<TAB>relation<TAB>tail<TAB>timestamp
    Files: train.txt, test.txt
    """
    def read_file(fname, upto):
        triples = []
        with open(fname, "r") as f:
            count=0
            for line in f:
                count += 1
                # if count == 100:
                #     break
                parts = line.strip().split("\t")
                # print(parts)
                # if len(parts) != 4:
                

                # # Skip empty/malformed lines
                #     continue
                h, r, t, ts = parts[0], parts[1], parts[2], parts[3]
                if int(ts) >= upto :
                    continue
                triples.append((int(h), int(t), int(r), int(ts)))
        return torch.tensor(triples, dtype=torch.long)
    full_train = read_file(os.path.join(data_dir, "train.txt") ,100)
    test = read_file(os.path.join(data_dir, "test.txt"),4360)

    # Split train into train/valid by time
    # Sort by timestamp (column 3)
    sorted_train, indices = torch.sort(full_train[:, 3])
    full_train = full_train[indices]
    split_idx = int(len(full_train) * (1 - valid_ratio))
    train = torch.utils.data.Subset(full_train, range(0, split_idx))
    valid = torch.utils.data.Subset(full_train, range(split_idx, len(full_train)))

    # Compute relation count directly
    print(f"Full train size: {full_train.size()}, Test size: {test.size()}")

    all_relations = torch.cat([full_train[:,2], test[:,2]])
    num_relation = int(torch.max(all_relations)) + 1

    ds = {
     "train": full_train[train.indices],
"valid": full_train[valid.indices],
        "test": test
    }
    return ds , num_relation



class TemporalKGBenchmark(Dataset):
    def __init__(self, ds, split="train"):
        """
        ds: dict with keys 'train', 'valid', 'test', each containing a torch.LongTensor of shape (num_triples, 4)
        split: which split to use for this dataset object
        """
        self.data = ds[split]
        self.split = split

        # Combine all splits to compute total number of entities and relations
        all_facts = torch.cat([ds["train"], ds["valid"], ds["test"]], dim=0)

        # Compute number of entities and relations from the data
        self.num_entity = int(torch.max(torch.cat([all_facts[:,0], all_facts[:,1]])) + 1)
        self.num_relation = int(torch.max(all_facts[:,2]) + 1)
        print (self.data)
        # Create edge list and edge weights from current split
        h = self.data[:, 0]
        t = self.data[:, 1]
        r = self.data[:, 2]
        edge_list = torch.stack([h, t, r], dim=-1)
        edge_weight = torch.ones(edge_list.size(0))

        # Create graph object
        self.graph = td_data.Graph(
            edge_list=edge_list,
            edge_weight=edge_weight,
            num_node=self.num_entity,
            num_relation=self.num_relation
        )

        # Assign edge times
        with self.graph.edge():
            self.graph.edge_time = self.data[:, 3].clone()

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_quadruples(batch):
    return torch.stack(batch, dim=0)


# ---------------------------
# Train / Eval
# ---------------------------
def main(data_dir="data/ICEWS14",
         hidden_dims=(256, 256), message_func="distmult",
         time_encode_dim=4, time_decay="exp", time_half_life=64.0, time_window=None,
         batch_size=64, lr=1e-3, max_epoch=2,
         device="cuda" if torch.cuda.is_available() else "cpu"):


    print("[DEBUG] Starting data loading...")


    ds, num_relation= load_local_icews(data_dir)
    train_dataset = TemporalKGBenchmark(ds, split="train")
    valid_dataset = TemporalKGBenchmark(ds,  split="valid")
    test_dataset = TemporalKGBenchmark(ds, split="test")
    # print(f"[DEBUG] Train: {len(dataset.train)}, Valid: {len(dataset.valid)}, Test: {len(dataset.test)}")

 
    print(f"[DEBUG] Train set size: {len(train_dataset)}, Valid set size: {len(valid_dataset)}, Test set size: {len(test_dataset)}")
    print("[DEBUG] Initializing model...")
    model = NeuralBellmanFordNetworkTemporal(
        input_dim=4,
        hidden_dims=[4, 4, 4, 4, 4, 4],
        num_relation=num_relation,
        message_func=message_func,
        aggregate_func="pna",
        short_cut=True,
        layer_norm=True,
        activation="relu",
        concat_hidden=False,
        num_mlp_layer=2,
        dependent=True,
        remove_one_hop=False,
        time_encode_dim=time_encode_dim,
        time_decay=time_decay,
        time_half_life=time_half_life,
        time_window=time_window,
    ).to(device) # <-- Move model to GPU

    print(f"[DEBUG] Initialized model with {sum(p.numel() for p in model.parameters())} parameters.")

    print("[DEBUG] Creating task and moving to device...")
    task = TemporalKnowledgeGraphCompletion(
        model,
        criterion="bce",
        metric=("mr", "mrr", "hits@1", "hits@3", "hits@10"),
        
        num_negative=31,
        strict_negative=True,
        filtered_ranking=True,
        full_batch_eval=True,
    )# <-- Move task to GPU

    print("[DEBUG] Creating engine...")
    optimizer = torch.optim.Adam(task.parameters(), lr=lr)
    engine = core.Engine(
        task,
        train_set=train_dataset,
        valid_set=valid_dataset,
        test_set=test_dataset,
        batch_size=batch_size,
        gpus=[0],
        optimizer=optimizer
    )

    print("[DEBUG] Starting training loop...")
    for epoch in range(max_epoch):
        print(f"[DEBUG] Epoch {epoch} begin")
        engine.train()
        print(f"[DEBUG] Epoch {epoch} training finished, starting evaluation...")
        results = engine.evaluate("valid")
        print(f"[DEBUG] Epoch {epoch}: Validation results: {results}")

    print("[DEBUG] Training finished. Evaluating on test set...")
    test_results = engine.evaluate("test")
    print(f"[DEBUG] Test results: {test_results}")


if __name__ == "__main__":
    main()
