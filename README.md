# Hex Game Engine (C++)

This project provides an object-oriented implementation of the board game Hex. The engine models the board, game state, coordinate systems, and winner detection in a clear and extensible architecture. It serves as a foundation for future AI methods such as Monte Carlo Tree Search and neural network evaluation (AlphaZero style).

## Game Overview

Hex is a deterministic two-player connection game played on an N×N rhombus-shaped board of hexagonal cells.

- Player X aims to connect the left and right sides of the board.
- Player O aims to connect the top and bottom sides.
- Draws are theoretically impossible; one player must win.

This project currently supports human-vs-human gameplay through a text-based interface.

## Object-Oriented Structure

| Class | Description |
|------|-------------|
| `Board` | Maintains the board matrix, handles move placement, and renders the board to the console. |
| `GameState` | Wraps the board as a logical state, tracks the current player, generates legal moves, and determines if the game is finished. |
| `Cube` | Represents positions using cube coordinates, enabling robust neighbor calculations for winner detection. |
| `MoveStrategy` | Defines interfaces and implementations for move selection (Random, Monte Carlo, Negamax scaffold). |
| `Zobrist` | Provides hashing utilities for incremental state hashing (for transposition tables). |
| `gnn::Graph` | Represents the Hex board as a graph with per-node features, ready for GNN input. |
| `FeatureExtractor` | Converts `Board`/`GameState` into flattened tensors (features + edge lists) for the GNN backend. |

The separation of state logic from board representation makes this implementation suitable for integration with search algorithms and machine learning components.

## Coordinate System

Although the board is stored in a standard 2D matrix (row, column), winner detection uses cube coordinates. Cube coordinates allow each cell to have exactly six neighbors, simplifying adjacency checks.

The system converts each board cell to cube coordinates, enabling graph search operations that are independent of the visual representation.

## Winner Detection (BFS)

Winner detection proceeds by:

1. Identifying starting edge cells for a given player.
2. Mapping all board cells to cube coordinates.
3. Performing a Breadth-First Search (BFS) through connected stones of the same player.
4. Detecting whether the search reaches the opposite target edge.

If so, the current player is declared the winner.

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
./hex

## Self-Play Data Generation

A separate `selfplay` target generates training data for a future GNN evaluator:

```bash
cd selfplay
mkdir build && cd build
cmake ..
make
./selfplay <games> <sims-base> <output.jsonl>
```

- Runs self-play across board sizes 4..11.
- Per game, Monte Carlo simulations per move are randomized in [4, 20].
- Writes one JSONL file per board size with samples `(N, board, to_move, result)`.

## Current Progress

- Core engine (Board, GameState, Cube) implemented with winner detection via BFS.
- Move strategies: Random, Monte Carlo; Negamax scaffold with Zobrist hashing utilities started.
- Graph representation and feature extraction for GNN input implemented in C++.
- Self-play generator produces training data across multiple board sizes with randomized simulations.

## Next Steps

- Implement GNNModel loading/inference (e.g., ONNX/LibTorch) and integrate value evaluation into search.
- Flesh out Negamax (transposition table, move ordering, time controls) and add a proper evaluation function.
- Expand self-play options (strategy mix, more control over simulations/time) and add validation scripts.
