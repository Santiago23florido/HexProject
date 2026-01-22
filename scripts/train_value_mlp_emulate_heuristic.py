#!/usr/bin/env python3
# scripts/train_value_mlp_emulate_heuristic.py

import os
import math
import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# 1) Target: "segunda dinámica"
# -----------------------------
def heuristic_score_second_dynamics(
    dist_self: torch.Tensor,
    dist_opp: torch.Tensor,
    libs_self: torch.Tensor,
    libs_opp: torch.Tensor,
    bridges_self: torch.Tensor,
    bridges_opp: torch.Tensor,
    center: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """
    Replica exactamente la parte final de tu heuristicEval (sin los returns dist==0).
    Todos los tensores deben ser float32 y de shape [B].
    """
    pathWeight = 600 + N * 10
    threatWeight = 12000 + N * 300
    bridgeWeight = 6 + N
    libertyWeight = 2
    centerWeight = 1

    score = torch.zeros_like(dist_self)

    score = score + (dist_opp - dist_self) * pathWeight
    score = score + (dist_self == 1.0).to(score.dtype) * threatWeight
    score = score - (dist_opp == 1.0).to(score.dtype) * threatWeight
    score = score + (bridges_self - bridges_opp) * bridgeWeight
    score = score + (libs_self - libs_opp) * libertyWeight
    score = score + center * centerWeight

    return score


# -----------------------------
# 2) Sampling de features
# -----------------------------
@dataclass
class FeatureConfig:
    N: int = 11
    p_dist_is_one: float = 0.12  # prob. de dist==1 para aprender los "if"
    # Rangos sintéticos (ajustables si quieres aproximarte más a tus rangos reales)
    libs_max_mult: int = 6       # libs en [0, 6*N*N]
    bridges_max_mult: int = 2    # bridges en [0, 2*N*N]
    center_max_mult: int = 1     # center en [-1*N*N, +1*N*N]


def sample_features(batch_size: int, cfg: FeatureConfig, device: torch.device):
    """
    Devuelve:
      x: [B, F] float32
      y: [B] float32 (score objetivo)
    Inputs "mismas entradas" de tu heurística:
      distSelf, distOpp, libsSelf, libsOpp, bridgesSelf, bridgesOpp, center
    """
    N = cfg.N
    maxDist = max(1, N * N)

    # dist in [1, maxDist] + forzamos algunos a 1 para aprender threatWeight
    dist_self = torch.randint(1, maxDist + 1, (batch_size,), device=device)
    dist_opp = torch.randint(1, maxDist + 1, (batch_size,), device=device)

    mask_s = torch.rand(batch_size, device=device) < cfg.p_dist_is_one
    mask_o = torch.rand(batch_size, device=device) < cfg.p_dist_is_one
    dist_self[mask_s] = 1
    dist_opp[mask_o] = 1

    libs_max = cfg.libs_max_mult * maxDist
    bridges_max = cfg.bridges_max_mult * maxDist
    center_max = cfg.center_max_mult * maxDist

    libs_self = torch.randint(0, libs_max + 1, (batch_size,), device=device)
    libs_opp = torch.randint(0, libs_max + 1, (batch_size,), device=device)

    bridges_self = torch.randint(0, bridges_max + 1, (batch_size,), device=device)
    bridges_opp = torch.randint(0, bridges_max + 1, (batch_size,), device=device)

    center = torch.randint(-center_max, center_max + 1, (batch_size,), device=device)

    # Cast a float32
    dist_self = dist_self.to(torch.float32)
    dist_opp = dist_opp.to(torch.float32)
    libs_self = libs_self.to(torch.float32)
    libs_opp = libs_opp.to(torch.float32)
    bridges_self = bridges_self.to(torch.float32)
    bridges_opp = bridges_opp.to(torch.float32)
    center = center.to(torch.float32)

    y = heuristic_score_second_dynamics(
        dist_self, dist_opp, libs_self, libs_opp, bridges_self, bridges_opp, center, N
    )

    # Feature vector con EXACTAMENTE las mismas entradas que usa la ecuación
    x = torch.stack(
        [dist_self, dist_opp, libs_self, libs_opp, bridges_self, bridges_opp, center],
        dim=1,
    )
    return x, y


# -----------------------------
# 3) Modelo (MLP) + Normalización
# -----------------------------
class FeatureNorm(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class HexValueMLP(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, hidden: int, depth: int, target_scale: float):
        super().__init__()
        self.norm = FeatureNorm(mean, std)
        self.target_scale = float(target_scale)  # para export reproducible

        layers = []
        in_dim = mean.numel()
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.core = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, F]
        return: [B] score (misma escala que la heurística)
        """
        z = self.norm(x)
        out_scaled = self.core(z).squeeze(1)        # predice score/target_scale
        out = out_scaled * self.target_scale
        return out


# -----------------------------
# 4) Entrenamiento
# -----------------------------
def estimate_norm(cfg: FeatureConfig, device: torch.device, num_batches: int = 200, batch_size: int = 2048):
    """
    Estima mean/std de features muestreando datos sintéticos.
    """
    xs = []
    for _ in range(num_batches):
        x, _ = sample_features(batch_size, cfg, device)
        xs.append(x)
    X = torch.cat(xs, dim=0)
    mean = X.mean(dim=0)
    std = X.std(dim=0).clamp_min(1e-6)
    return mean, std


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--target-scale", type=float, default=10000.0,
                    help="La red predice score/scale y luego se multiplica por scale para volver al score real.")
    ap.add_argument("--p-dist-is-one", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="scripts/models/hex_value_ts_mp.pt")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seeds(args.seed)
    device = torch.device(args.device)

    cfg = FeatureConfig(
        N=args.N,
        p_dist_is_one=args.p_dist_is_one,
    )

    # Normalización de features
    mean, std = estimate_norm(cfg, device=device)
    model = HexValueMLP(mean=mean, std=std, hidden=args.hidden, depth=args.depth, target_scale=args.target_scale).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Entrenamiento
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for step in range(args.steps_per_epoch):
            x, y = sample_features(args.batch_size, cfg, device)

            # Entrenamos contra targets escalados para estabilidad
            y_scaled = y / args.target_scale

            pred = model(x)  # pred en escala real
            pred_scaled = pred / args.target_scale

            loss = loss_fn(pred_scaled, y_scaled)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()

        avg = running / args.steps_per_epoch
        print(f"Epoch {epoch:02d} | MSE(scaled) {avg:.6f}")

    # Validación rápida
    model.eval()
    with torch.no_grad():
        x, y = sample_features(8192, cfg, device)
        pred = model(x)
        mae = (pred - y).abs().mean().item()
        rmse = torch.sqrt(((pred - y) ** 2).mean()).item()
        print(f"Quick eval | MAE {mae:.2f} | RMSE {rmse:.2f}")

    # Export TorchScript
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Script (si falla por algo, puedes cambiar a trace)
    scripted = torch.jit.script(model.cpu())
    scripted.save(out_path)
    print(f"Saved TorchScript model to: {out_path}")

    # Smoke test load
    loaded = torch.jit.load(out_path)
    loaded.eval()
    with torch.no_grad():
        x, y = sample_features(4, cfg, device=torch.device("cpu"))
        p = loaded(x)
        print("Smoke test preds:", p.tolist())
        print("Smoke test true :", y.tolist())


if __name__ == "__main__":
    main()
