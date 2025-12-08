# Hex Project

Hex game engine and AI stack in C++ with a TorchScript GNN value network. The repository contains:
- A full Hex rules implementation (board, game state, winner detection).
- Two search-based agents: a handcrafted heuristic Negamax and a GNN-guided Negamax.
- A self-play generator to create training data for the GNN.
- A Python training script to produce the TorchScript model consumed by C++.

## Engine and Rules
- Board and state: `Board`, `GameState`, `Cube` model the 7×7 Hex board (configurable) with cube coordinates for robust neighbor lookup.
- Winner detection: BFS over connected stones reaches opposite sides (X: left–right, O: top–bottom).
- Move legality and turn tracking handled by `GameState`.

## Agents
- **Heuristic Negamax**: evaluation mixes shortest-path heuristics, liberties, bridges, center control, stone count, and immediate-win checks. Hashing via Zobrist plus a transposition table.
- **GNN Negamax**: same search scaffold, but leaf evaluation comes from a TorchScript model (`scripts/models/hex_value_ts.pt`). Current defaults: depth 5, 3000 ms per move for GNN; depth 3, 2000 ms for heuristic.
- **Interactive play**: `./build/hex` asks if you want heuristic (`h`, default) or GNN (`g`).

## Build and Run
```bash
cmake -S . -B build
cmake --build build
./build/hex
```

## Self-Play Generator
Binary under `selfplay/` to produce JSONL training data.

Build:
```bash
cmake -S selfplay -B selfplay/build
cmake --build selfplay/build
```

Run:
```bash
./selfplay/build/selfplay <games> <minDepth> <maxDepth> <outputPath> <minPairs> <maxPairs> <timeLimitMs>
```
- Players: two heuristic Negamax agents with depth randomized in `[minDepth, maxDepth]`.
- Time per move: `timeLimitMs` ms.
- Starting positions: random pairs of stones per player (`pairs ∈ [minPairs, maxPairs]`), rejecting boards unless both players have at least one connected chain of length ≥ 2 to ensure meaningful mid/late-game patterns.
- Files: writes one JSONL per board size, e.g. `selfplay_data_N7.jsonl`. Flushes every 20 games to avoid high RAM use.
- Sample fields: `N`, `board` (flattened), `to_move`, `result` (+1/-1/0 from `to_move` perspective), `moves_to_end` (plies remaining).

## GNN Features and Training (Python)
Script: `scripts/train_gnn.py`.
- Input features per node (10): `p1, p2, empty, sideA, sideB, degree, distToA, distToB, toMoveP1, toMoveP2` — identical ordering in Python and C++ extractors. Distances are BFS hops to the target borders, normalized by board area, computed without supernodes.
- Model: 3 message-passing blocks (hidden 128) with mean aggregation, layer-norm residuals, global mean+max pooling, and two heads.
- Outputs:
  - Value head: scalar in `[-1, 1]` meaning advantage for the *current player* (TorchScript and C++ wrapper use the same sign; no flips).
  - Auxiliary head: predicts normalized `moves_to_end` (training only).
- Loss: SmoothL1 on value targets (`result` clamped to [-1,1]) scaled by a confidence weight `w = 0.2 + 0.8*(1 - moves_norm)`, optionally amplified by `endgame_weight`. Aux MSE is added with `--aux-weight`.
- Data shuffle each epoch; dataset stats (mean/std, +/- counts, examples) printed once for sanity. `--self-test` runs a tiny 2-sample overfit check to verify the pipeline.
- Arguments: `--epochs`, `--lr`, `--aux-weight`, `--endgame-weight`, `--data`, `--limit`, `--output`, `--self-test`.

Example training:
```bash
python3 scripts/train_gnn.py \
  --data selfplay/build/selfplay_data_N7.jsonl \
  --epochs 20 --lr 1e-3 \
  --aux-weight 0.1 --endgame-weight 1.0
```
The TorchScript model is saved to `scripts/models/hex_value_ts.pt` and loaded automatically by `./build/hex` and the self-play generator.

## Code Map (high level)
- `src/`: game loop (`main.cpp`), strategies, hashing, GNN wrapper.
- `include/`: headers for board/state/strategies/gnn.
- `selfplay/`: standalone generator and serializer for JSONL data.
- `scripts/`: training script and model artifacts.

## Notes and Next Steps
- Improve search ordering (killer/history heuristics), and tune depth/time defaults.
- Expand self-play curriculum (more sizes, teacher-student setups).
- Add symmetry augmentation during training to multiply examples.
- Optional: integrate policy head to guide move ordering alongside value.
