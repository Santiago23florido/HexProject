# Architecture Overview

This page mirrors the project structure diagram from the report, with README and docs added. Build folders are intentionally omitted, and only source files (`.cpp`, `.hpp`), Python scripts (`.py`), and trained models are listed.

```
HexProject/
|-- README.md
|-- docs/
|-- scripts/
|   |-- models/
|   |   |-- hex_value_ts_mp.pt
|   |   |-- hex_value.pt
|   |   |-- hex_value_ts.pt
|   |   `-- value_mlp_state.pt
|   |-- __pycache__/
|   `-- train_value_mlp_emulate_heuristic.py
|-- include/
|   |-- core/
|   |   |-- Board.hpp
|   |   |-- Cube.hpp
|   |   |-- GameState.hpp
|   |   |-- MoveStrategy.hpp
|   |   `-- Player.hpp
|   |-- gnn/
|   |   |-- FeatureExtractor.hpp
|   |   |-- GNNModel.hpp
|   |   `-- Graph.hpp
|   `-- ui/
|       |-- HexGameUI.hpp
|       |-- HexTile.hpp
|       `-- ImageViewer.hpp
|-- src/
|   |-- core/
|   |   |-- Board.cpp
|   |   |-- Cube.cpp
|   |   |-- GameState.cpp
|   |   |-- MoveStrategy.cpp
|   |   `-- Player.cpp
|   |-- gnn/
|   |   |-- FeatureExtractor.cpp
|   |   |-- GNNModel.cpp
|   |   `-- Graph.cpp
|   |-- ui/
|   |   |-- HexGameUI.cpp
|   |   |-- HexGameUI_updateVolumeIcon.cpp
|   |   |-- HexTile.cpp
|   |   |-- ImageViewer.cpp
|   |   `-- main.cpp
|   `-- cli/
|       `-- main.cpp
`-- selfplay/
    |-- gnn/
    |   |-- DataCollector.cpp
    |   |-- DataCollector.hpp
    |   |-- GameRunner.cpp
    |   |-- GameRunner.hpp
    |   |-- Serializer.cpp
    |   |-- Serializer.hpp
    |   `-- maingnn.cpp
    |-- mlp/
    |   |-- ReplayBuffer.cpp
    |   |-- ReplayBuffer.hpp
    |   |-- RLTrainer.cpp
    |   |-- RLTrainer.hpp
    |   |-- ValueMLP.cpp
    |   `-- ValueMLP.hpp
    `-- main.cpp
```

## Classes and Structures
| Class/Structure | Module |
| --- | --- |
| Board, GameState, Player, IMoveStrategy | core engine (`src/core`, `include/core`) |
| Graph, FeatureExtractor, GNNModel | GNN utilities (`src/gnn`, `include/gnn`) |
| RLTrainer, ValueMLP, ReplayBuffer | self‑play MLP training (`selfplay/mlp`) |
| DataCollector, GameRunner, Serializer | self‑play data generation (`selfplay/gnn`) |
| HexGameUI, HexTile, ImageViewer | UI (`src/ui`, `include/ui`) |
