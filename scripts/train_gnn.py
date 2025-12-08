import argparse
import glob
import json
import math
import os
import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# Neighbor deltas for odd-r offset coordinates
EVEN_NEIGHBORS = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
ODD_NEIGHBORS = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]


def load_samples(paths: List[str], limit: int = None) -> List[dict]:
    samples = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                samples.append(json.loads(line))
                if limit is not None and len(samples) >= limit:
                    return samples
    return samples


def select_default_data() -> List[str]:
    """
    Pick the most recent N=7 JSONL self-play file under selfplay/build.
    Falls back to the newest JSONL if none match the _N7 pattern.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.normpath(os.path.join(here, "..", "selfplay", "build"))
    paths_n7 = glob.glob(os.path.join(default_dir, "*_N7.jsonl"))
    candidates = paths_n7 if paths_n7 else glob.glob(os.path.join(default_dir, "*.jsonl"))
    if not candidates:
        return []
    newest = max(candidates, key=os.path.getmtime)
    return [newest]


def build_graph(sample: dict) -> torch.Tensor:
    """
    Build edge index for the hex grid (no supernodes in edge list).
    Returns edge_index of shape [2, E].
    """
    N = sample["N"]
    board_len = N * N
    edges = []
    for r in range(N):
        for c in range(N):
            idx = r * N + c
            deltas = EVEN_NEIGHBORS if (r % 2 == 0) else ODD_NEIGHBORS
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N:
                    nidx = nr * N + nc
                    edges.append((idx, nidx))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]
    return edge_index


def bfs_distances(N: int, edge_index: torch.Tensor, targets: List[int]) -> List[float]:
    """
    Unweighted BFS to compute shortest-hop distances from a set of target nodes.
    Returns a list of size N*N with normalized distances (divided by N*N).
    """
    num_nodes = N * N
    neighbors = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        neighbors[s].append(d)
    dist = [math.inf] * num_nodes
    queue = []
    for t in targets:
        dist[t] = 0.0
        queue.append(t)
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        du = dist[u]
        for v in neighbors[u]:
            if dist[v] > du + 1:
                dist[v] = du + 1
                queue.append(v)
    norm = float(num_nodes if num_nodes > 0 else 1)
    return [0.0 if math.isinf(d) else d / norm for d in dist]


def build_features(sample: dict, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Build per-node features matching C++ FeatureExtractor plus to-move:
    p1, p2, empty, sideA, sideB, degree, distToA, distToB, toMoveP1, toMoveP2.
    """
    N = sample["N"]
    board = sample["board"]
    to_move = sample.get("to_move", 1)
    num_nodes = N * N
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    for s in edge_index[0].tolist():
        degree[s] += 1.0
    degree = degree / 6.0  # normalize

    # Targets for BFS distances (approximate supernodes)
    targetsA = [r * N for r in range(N)] + [r * N + (N - 1) for r in range(N)]  # columns
    targetsB = list(range(N)) + list(range((N - 1) * N, N * N))  # rows
    dist_to_a = bfs_distances(N, edge_index, targetsA)
    dist_to_b = bfs_distances(N, edge_index, targetsB)

    feats = torch.zeros((num_nodes, 10), dtype=torch.float32)
    for idx in range(num_nodes):
        val = board[idx]
        r = idx // N
        c = idx % N
        feats[idx, 0] = 1.0 if val == 1 else 0.0  # p1
        feats[idx, 1] = 1.0 if val == 2 else 0.0  # p2
        feats[idx, 2] = 1.0 if val == 0 else 0.0  # empty
        feats[idx, 3] = 1.0 if (c == 0 or c == N - 1) else 0.0  # sideA
        feats[idx, 4] = 1.0 if (r == 0 or r == N - 1) else 0.0  # sideB
        feats[idx, 5] = degree[idx]
        feats[idx, 6] = dist_to_a[idx]
        feats[idx, 7] = dist_to_b[idx]
        feats[idx, 8] = 1.0 if to_move == 1 else 0.0  # toMoveP1
        feats[idx, 9] = 1.0 if to_move == 2 else 0.0  # toMoveP2
    return feats


def compute_weight(moves_norm: Optional[float], endgame_weight: float) -> float:
    conf = 1.0 - moves_norm if moves_norm is not None else 1.0
    w = 0.2 + 0.8 * conf
    if endgame_weight > 0:
        w *= (1.0 + endgame_weight * conf)
    return w


def prepare_targets(samples: List[dict], endgame_weight: float):
    targets_cache = []
    target_vals = []
    pos_ones = 0
    neg_ones = 0
    for s in samples:
        raw_result = float(s["result"])
        target_val = max(-1.0, min(1.0, raw_result))
        moves_to_end = s.get("moves_to_end", None)
        moves_norm = None
        if moves_to_end is not None:
            moves_norm = float(moves_to_end) / float(max(1, s["N"] * s["N"]))
            moves_norm = max(0.0, min(1.0, moves_norm))
        weight = compute_weight(moves_norm, endgame_weight)

        targets_cache.append(
            {
                "target_val": target_val,
                "moves_norm": moves_norm,
                "weight": weight,
            }
        )
        target_vals.append(target_val)
        if target_val >= 0.999:
            pos_ones += 1
        if target_val <= -0.999:
            neg_ones += 1

    count = max(1, len(target_vals))
    mean_val = sum(target_vals) / count
    var_val = sum((v - mean_val) ** 2 for v in target_vals) / count
    std_val = math.sqrt(var_val)
    stats = {
        "count": len(target_vals),
        "mean": mean_val,
        "std": std_val,
        "pos_ones": pos_ones,
        "neg_ones": neg_ones,
    }
    return targets_cache, stats


def log_dataset_stats(samples: List[dict], targets_cache: List[dict], stats: dict):
    print(
        f"[Data] target_val mean={stats['mean']:.4f} std={stats['std']:.4f} "
        f"+1={stats['pos_ones']} -1={stats['neg_ones']}"
    )
    for i in range(min(3, len(samples))):
        to_move = samples[i].get("to_move", 1)
        result = samples[i].get("result")
        target_val = targets_cache[i]["target_val"]
        print(f"[Data] example {i}: to_move={to_move} result={result} target_val={target_val:.3f}")
    if stats["count"] > 0:
        skewed = stats["pos_ones"] == 0 or stats["neg_ones"] == 0
        min_bucket = min(stats["pos_ones"], stats["neg_ones"])
        if skewed or (min_bucket / float(stats["count"]) < 0.05):
            print("[Data][warn] target_val distribution is highly skewed; check self-play balance.")


def resolve_data_paths(args) -> List[str]:
    data_paths = []
    if args.data:
        existing = [p for p in args.data if os.path.exists(p)]
        missing = [p for p in args.data if not os.path.exists(p)]
        if missing:
            print(f"Warning: missing data files skipped: {missing}")
        data_paths = existing
    if not data_paths:
        data_paths = select_default_data()
    return data_paths


class MessagePassingBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_neigh = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]

        # Learned transform on neighbor messages, mean-aggregated
        neigh = torch.zeros_like(h)
        neigh.index_add_(0, src, self.lin_neigh(h[dst]))
        deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        deg.index_add_(0, src, torch.ones_like(src, device=h.device, dtype=h.dtype))
        deg = torch.clamp(deg, min=1.0)
        neigh = neigh / deg.unsqueeze(-1)

        # Self update + residual + normalization
        out = self.lin_self(h) + neigh
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.norm(h + out)
        return out


class SimpleGNN(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 128, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([MessagePassingBlock(hidden, dropout) for _ in range(num_layers)])
        pooled_dim = hidden * 2  # mean + max pooling
        self.head_value = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.head_moves = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_and_pool(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.global_pool(h)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

    def global_pool(self, h: torch.Tensor) -> torch.Tensor:
        mean = h.mean(dim=0, keepdim=True)
        max_pool = torch.max(h, dim=0, keepdim=True).values
        return torch.cat([mean, max_pool], dim=-1)

    def value_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        logits = self.head_value(pooled)
        prob = torch.sigmoid(logits)
        val = prob * 2.0 - 1.0
        return val.squeeze()

    def moves_from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        moves = torch.relu(self.head_moves(pooled))
        return moves.squeeze()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_and_pool(x, edge_index)
        return self.value_from_pooled(pooled)

    def predict_moves(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_and_pool(x, edge_index)
        return self.moves_from_pooled(pooled)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_paths = resolve_data_paths(args)
    if not data_paths:
        raise RuntimeError("No data files found. Generate self-play JSONL under selfplay/build/ or pass --data.")

    samples = load_samples(data_paths, limit=args.limit)
    print(f"Loaded {len(samples)} samples from {data_paths}")
    if not samples:
        raise RuntimeError("Dataset is empty after loading samples.")

    targets_cache, stats = prepare_targets(samples, args.endgame_weight)
    log_dataset_stats(samples, targets_cache, stats)

    model = SimpleGNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.1)
    huber = nn.SmoothL1Loss(reduction="none")
    aux_loss_fn = nn.MSELoss()

    indices = list(range(len(samples)))
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        random.shuffle(indices)
        for idx in indices:
            s = samples[idx]
            edge_index = build_graph(s).to(device)
            feats = build_features(s, edge_index.cpu()).to(device)

            pooled = model.encode_and_pool(feats, edge_index)
            pred_val = model.value_from_pooled(pooled)

            target_info = targets_cache[idx]
            target_tensor = torch.tensor(target_info["target_val"], device=device)
            weight_tensor = torch.tensor(target_info["weight"], device=device)
            loss_value = huber(pred_val, target_tensor) * weight_tensor
            loss_total = loss_value.mean()

            moves_norm = target_info["moves_norm"]
            if moves_norm is not None:
                moves_tensor = torch.tensor(moves_norm, device=device)
                pred_moves = model.moves_from_pooled(pooled)
                loss_aux = aux_loss_fn(pred_moves, moves_tensor)
                loss_total = loss_total + args.aux_weight * loss_aux

            opt.zero_grad()
            loss_total.backward()
            opt.step()

            total_loss += loss_total.item()
        avg_loss = total_loss / max(1, len(samples))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")
        scheduler.step()

    if args.output:
        out_path = args.output
        # Resolve relative paths from the script directory so C++ can load scripts/models/hex_value_ts.pt
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        model_cpu = model.cpu()
        model_cpu.eval()
        scripted = torch.jit.script(model_cpu)
        scripted.save(out_path)
        print(f"Saved TorchScript model to {out_path}")


def self_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SelfTest] Using device: {device}")

    data_paths = resolve_data_paths(args)
    if not data_paths:
        raise RuntimeError("No data files found for self-test.")
    samples = load_samples(data_paths, limit=max(args.limit or 2, 2))
    if len(samples) < 2:
        raise RuntimeError("Self-test requires at least 2 samples.")

    targets_cache, stats = prepare_targets(samples, args.endgame_weight)
    log_dataset_stats(samples, targets_cache, stats)

    model = SimpleGNN().to(device)
    huber = nn.SmoothL1Loss(reduction="none")
    aux_loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    subset = list(range(min(2, len(samples))))

    def eval_subset_loss():
        total = 0.0
        with torch.no_grad():
            for idx in subset:
                s = samples[idx]
                edge_index = build_graph(s).to(device)
                feats = build_features(s, edge_index.cpu()).to(device)
                pooled = model.encode_and_pool(feats, edge_index)

                target_info = targets_cache[idx]
                pred_val = model.value_from_pooled(pooled)
                target_tensor = torch.tensor(target_info["target_val"], device=device)
                weight_tensor = torch.tensor(target_info["weight"], device=device)
                loss_value = huber(pred_val, target_tensor) * weight_tensor
                loss_total = loss_value.mean()

                moves_norm = target_info["moves_norm"]
                if moves_norm is not None:
                    moves_tensor = torch.tensor(moves_norm, device=device)
                    pred_moves = model.moves_from_pooled(pooled)
                    loss_aux = aux_loss_fn(pred_moves, moves_tensor)
                    loss_total = loss_total + args.aux_weight * loss_aux
                total += loss_total.item()
        return total / max(1, len(subset))

    initial_loss = eval_subset_loss()
    steps = max(1, min(20, args.epochs))
    for _ in range(steps):
        for idx in subset:
            s = samples[idx]
            edge_index = build_graph(s).to(device)
            feats = build_features(s, edge_index.cpu()).to(device)
            pooled = model.encode_and_pool(feats, edge_index)

            target_info = targets_cache[idx]
            pred_val = model.value_from_pooled(pooled)
            target_tensor = torch.tensor(target_info["target_val"], device=device)
            weight_tensor = torch.tensor(target_info["weight"], device=device)
            loss_value = huber(pred_val, target_tensor) * weight_tensor
            loss_total = loss_value.mean()

            moves_norm = target_info["moves_norm"]
            if moves_norm is not None:
                moves_tensor = torch.tensor(moves_norm, device=device)
                pred_moves = model.moves_from_pooled(pooled)
                loss_aux = aux_loss_fn(pred_moves, moves_tensor)
                loss_total = loss_total + args.aux_weight * loss_aux

            opt.zero_grad()
            loss_total.backward()
            opt.step()

    final_loss = eval_subset_loss()
    print(f"[SelfTest] loss start={initial_loss:.4f} end={final_loss:.4f} over {steps} steps")
    if final_loss >= initial_loss:
        print("[SelfTest][warn] Loss did not decrease; inspect data or hyperparameters.")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "models", "hex_value_ts.pt")
    default_data = os.path.normpath(os.path.join(here, "..", "selfplay", "build", "selfplay_data_N7.jsonl"))

    parser = argparse.ArgumentParser(description="Train a simple GNN value network for Hex.")
    parser.add_argument(
        "--data",
        nargs="+",
        default=[default_data],
        help="JSONL files with self-play samples (default: selfplay/build/selfplay_data_N7.jsonl, falls back to newest JSONL)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--aux-weight", type=float, default=0.1, help="Weight for moves_to_end auxiliary loss (default: 0.1)")
    parser.add_argument(
        "--endgame-weight",
        type=float,
        default=1.0,
        help="Extra weight for late-game samples (uses moves_to_end) (default: 1.0)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument(
        "--output",
        type=str,
        default=default_out,
        help="Path to save the TorchScript model (single file used by C++) (default: scripts/models/hex_value_ts.pt)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny overfit check on 2 samples to verify the pipeline instead of full training.",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test(args)
    else:
        train(args)
