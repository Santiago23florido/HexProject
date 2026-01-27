# Hex Project

Hex game engine and AI stack in C++ with a TorchScript value network (MLP by default, optional GNN). The repository contains:
- A full Hex rules implementation (board, game state, winner detection).
- Two search-based agents: a handcrafted heuristic Negamax and a TorchScript-guided Negamax.
- A self-play generator and trainer to build value data or export a TorchScript value model.
- A Python training script to produce the TorchScript model consumed by C++.

## Engine and Rules
- Board and state: `Board`, `GameState`, `Cube` model the 7×7 Hex board (configurable) with cube coordinates for robust neighbor lookup.
- Winner detection: BFS over connected stones reaches opposite sides (X: left–right, O: top–bottom).
- Move legality and turn tracking handled by `GameState`.

## Agents
- **Heuristic Negamax**: evaluation mixes shortest-path heuristics, liberties, bridges, center control, stone count, and immediate-win checks. Hashing via Zobrist plus a transposition table.
- **TorchScript Negamax**: same search scaffold, but leaf evaluation comes from a TorchScript model (`scripts/models/hex_value_ts_mp.pt`). The loader auto-detects whether the model expects graph inputs (GNN) or flat features (MLP). Current defaults: depth 5, 3000 ms per move for TorchScript; depth 3, 2000 ms for heuristic.
- **Interactive play**: `./build/hex` asks if you want heuristic (`h`, default) or TorchScript (`g`).

## Installation
### Prerequisites
CMake:
```bash
cmake --version
```

C++17 compiler:
```bash
g++ --version
# or: clang++ --version
```

LibTorch (TorchScript):
```bash
ls $HOME/libtorch/share/cmake/Torch
# or set Torch_DIR/CMAKE_PREFIX_PATH
```

SFML 2.5:
```bash
pkg-config --modversion sfml-all
# or: sudo apt-get install libsfml-dev
```

CUDA Toolkit (optional):
```bash
nvcc --version
```

Python 3 + poetry (optional):
```bash
python3 --version
poetry --version
```

### Build and run (game)
```bash
cmake -S . -B build
cmake --build build
cd build
./hex_ui
```

### Build and run (selfplay)
```bash
cmake -S selfplay -B selfplay/build
cmake --build selfplay/build
./selfplay/build/selfplay \
  <games> <minDepth> \
  <maxDepth> <outputPath> \
  <minPairs> <maxPairs> \
  <timeLimitMs>
# or training/export:
./selfplay/build/selfplay \
  --selfplay-train \
  --export-ts \
  --train-games 200 \
  --min-depth 10 \
  --max-depth 20 \
  --batch-size 256 \
  --updates-per-game 1 \
  --device cuda
```

## Self-Play Generator and Trainer
Binary under `selfplay/` can either produce JSONL training data or train a value MLP and export TorchScript.

Run (JSONL data):
```bash
./selfplay/build/selfplay \
  <games> <minDepth> \
  <maxDepth> <outputPath> \
  <minPairs> <maxPairs> \
  <timeLimitMs>
```
- Players: two heuristic Negamax agents with depth randomized in `[minDepth, maxDepth]`.
- Time per move: `timeLimitMs` ms.
- Starting positions: random pairs of stones per player (`pairs ∈ [minPairs, maxPairs]`), rejecting boards unless both players have at least one connected chain of length ≥ 2 to ensure meaningful mid/late-game patterns.
- Files: writes one JSONL per board size, e.g. `selfplay_data_N7.jsonl`. Flushes every 20 games to avoid high RAM use.
- Sample fields: `N`, `board` (flattened), `to_move`, `result` (+1/-1/0 from `to_move` perspective), `moves_to_end` (plies remaining).

Run (self-play training):
```bash
./selfplay/build/selfplay \
  --selfplay-train \
  --export-ts \
  --train-games 200 \
  --min-depth 10 \
  --max-depth 20 \
  --batch-size 256 \
  --updates-per-game 1 \
  --device cuda
```

## Value Model (TorchScript)
### Default MLP (heuristic emulator, Python)
Script: `scripts/train_value_mlp_emulate_heuristic.py`.
- Input features (7): `distSelf, distOpp, libsSelf, libsOpp, bridgesSelf, bridgesOpp, center` (same ordering as `computeValueFeatures` in C++).
- Targets: synthetic feature sampling; the target score mirrors the linear combination inside the heuristic (without the early `dist==0` return).
- Model: MLP with feature normalization, configurable depth/hidden, ReLU activations, and `target_scale` to keep outputs in heuristic score scale.
- Output: TorchScript saved to `scripts/models/hex_value_ts_mp.pt`. The C++ runtime treats this as an MLP (single-input `forward`).

### Self-play ValueMLP (C++)
Trainer: `selfplay/mlp/RLTrainer.cpp` and `selfplay/mlp/RLTrainer.hpp`.
- Data: self-play games; each position stores the same 7 features; target is win/loss for the side to move scaled by `valueScale`.
- Loss: SmoothL1; normalization estimated from the replay buffer; optional frozen snapshots for opponents.
- Export: use `--export-ts` in the self-play binary to write `scripts/models/hex_value_ts_mp.pt`.

### Optional GNN path
If you provide a TorchScript model with `forward(x, edge_index)`, the engine uses `FeatureExtractor` to build a graph and 10 node features: `p1, p2, empty, sideA, sideB, degree, distToA, distToB, toMoveP1, toMoveP2`. This repo does not currently include a Python GNN training script.

## Code Map (high level)
- `src/`: game loop (`main.cpp`), strategies, hashing, GNN wrapper.
- `include/`: headers for board/state/strategies/gnn.
- `selfplay/`: JSONL generator plus self-play trainer/exporter for the value MLP.
- `scripts/`: training script and model artifacts.

## Notes and Next Steps
- Improve search ordering (killer/history heuristics), and tune depth/time defaults.
- Expand self-play curriculum (more sizes, teacher-student setups).
- Add symmetry augmentation during training to multiply examples.
- Optional: integrate policy head to guide move ordering alongside value.
