import argparse
import json
import math
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import glob


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


def build_graph(sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge index for the hex grid (no supernodes in edge list).
    Returns edge_index of shape [2, E] and degree per node.
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
    Build per-node features matching C++ FeatureExtractor:
    p1, p2, empty, sideA, sideB, degree, distToA, distToB.
    """
    N = sample["N"]
    board = sample["board"]
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

    feats = torch.zeros((num_nodes, 8), dtype=torch.float32)
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
    return feats


class SimpleGNN(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 128):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, hidden)
        self.head_value = nn.Linear(hidden, 1)
        self.head_moves = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float32, device=x.device))
        deg = torch.clamp(deg, min=1.0)

        agg = torch.zeros_like(x)
        agg.index_add_(0, src, x[dst])
        agg = agg / deg.unsqueeze(-1)

        h = torch.relu(self.lin1(x + agg))
        h = torch.relu(self.lin2(h))
        h = torch.relu(self.lin3(h))
        h = h.mean(dim=0, keepdim=True)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        out = torch.tanh(self.head_value(h))  # value in [-1,1]
        return out.squeeze()

    def predict_moves(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        moves = torch.relu(self.head_moves(h))
        return moves.squeeze()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve data files: if none provided, use newest N=7 self-play JSONL under selfplay/build
    data_paths = args.data or select_default_data()
    if not data_paths:
        raise RuntimeError("No data files found. Generate self-play JSONL under selfplay/build/ or pass --data.")

    samples = load_samples(data_paths, limit=args.limit)
    print(f"Loaded {len(samples)} samples from {data_paths}")

    model = SimpleGNN().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    aux_loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for s in samples:
            edge_index = build_graph(s).to(device)
            feats = build_features(s, edge_index.cpu()).to(device)
            target = torch.tensor(float(s["result"]), device=device)
            moves_target = s.get("moves_to_end", None)
            moves_tensor = None
            if moves_target is not None:
                # Normalize by max plies (~ N*N) to keep target in [0,1]
                moves_tensor = torch.tensor(
                    float(moves_target) / float(max(1, s["N"] * s["N"])),
                    device=device,
                )

            opt.zero_grad()
            pred = model(feats, edge_index)
            loss = loss_fn(pred, target)
            if moves_tensor is not None:
                aux_pred = model.predict_moves(feats, edge_index)
                loss = loss + args.aux_weight * aux_loss_fn(aux_pred, moves_tensor)
            loss.backward()
            opt.step()

            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(samples))
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f}")

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


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "models", "hex_value_ts.pt")

    parser = argparse.ArgumentParser(description="Train a simple GNN value network for Hex.")
    parser.add_argument("--data", nargs="+", default=[], help="JSONL files with self-play samples (defaults to data/*.jsonl or selfplay/build/*.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--aux-weight", type=float, default=0.1, help="Weight for moves_to_end auxiliary loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default=default_out, help="Path to save the TorchScript model (single file used by C++)")
    args = parser.parse_args()
    train(args)
