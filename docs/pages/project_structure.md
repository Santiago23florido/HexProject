# Project Structure

The repository is organized to separate public headers (`include/`), C++ implementation (`src/`), self‑play tooling (`selfplay/`), and Python scripts and trained models (`scripts/`). The self‑play module is split into `gnn/` (data generation pipeline) and `mlp/` (training and value model export).

## Classes and Structures
| Class/Structure | Module |
| --- | --- |
| Board, GameState, Player, IMoveStrategy | core engine (`src/core`, `include/core`) |
| Graph, FeatureExtractor, GNNModel | GNN utilities (`src/gnn`, `include/gnn`) |
| RLTrainer, ValueMLP, ReplayBuffer | self‑play MLP training (`selfplay/mlp`) |
| DataCollector, GameRunner, Serializer | self‑play data generation (`selfplay/gnn`) |
| HexGameUI, HexTile, ImageViewer | UI (`src/ui`, `include/ui`) |
