# Object-Oriented Design (OOP)

The project applies OOP to separate responsibilities between game rules, AI strategies, and model evaluation. The table below summarizes how each concept is used and where it appears in the codebase.

## OOP Concepts and Usage
| Concept | How it is used | Where (examples) |
| --- | --- | --- |
| Modular class design | Core engine, GNN utilities, and self‑play are organized into dedicated modules. | `include/core/*`, `src/core/*`, `include/gnn/*`, `src/gnn/*`, `selfplay/*` |
| Inheritance | Common behavior is abstracted into base classes with specialized derived classes. | `Player` → `HumanPlayer`, `AIPlayer`, `HybridPlayer` (`include/core/Player.hpp`); `IMoveStrategy` → `RandomStrategy`, `MonteCarloStrategy`, `NegamaxStrategy` (`include/core/MoveStrategy.hpp`); `NegamaxStrategy` → `NegamaxHeuristicStrategy`, `NegamaxGnnStrategy` (`include/core/MoveStrategy.hpp`); `ValueMLPImpl` → `torch::nn::Module` (`selfplay/mlp/ValueMLP.hpp`) |
| Constructors | Default, parameterized, and copy constructors are used for safe object creation. | `Board()` in `src/cli/main.cpp`; `NegamaxStrategy(...)` in `selfplay/mlp/RLTrainer.cpp`; `Board` copy in `src/core/MoveStrategy.cpp` |
| Templates | Generic utilities are used for fixed arrays, range checks, and model evaluation dispatch. | `FixedArray`, `InRange`, `evalValueModel` (`include/core/MoveStrategy.hpp`) |
| Constraints | Template constraints restrict types to avoid misuse. | `InRange` uses `std::enable_if` for integral types (`include/core/MoveStrategy.hpp`) |
| Iterators | Iteration supports board traversal, BFS, and graph neighborhood access. | `GameState` iteration (`src/core/GameState.cpp`); graph adjacency traversal (`src/gnn/Graph.cpp`) |
| Operator overloads | Operators simplify coordinate math and container access. | `Cube::operator+` (`include/core/Cube.hpp`), `FixedArray::operator[]` (`include/core/MoveStrategy.hpp`), `HybridPlayer::operator=` (`include/core/Player.hpp`), GNN cache key equality/hash (`src/gnn/GNNModel.cpp`) |
| Exceptions | Runtime and input validation errors are surfaced explicitly. | Board/GameState size & bounds (`src/core/Board.cpp`, `src/core/GameState.cpp`); input failures (`src/cli/main.cpp`); model load/eval errors (`src/gnn/GNNModel.cpp`); CUDA availability (`selfplay/mlp/RLTrainer.cpp`); JSONL IO (`selfplay/gnn/Serializer.cpp`) |
| Smart pointers | Ownership and lifetime are made explicit with RAII. | `std::unique_ptr` in player strategies and self‑play (`include/core/Player.hpp`, `selfplay/mlp/RLTrainer.cpp`); `std::shared_ptr` PIMPL/cache in `GNNModel` (`include/gnn/GNNModel.hpp`, `src/gnn/GNNModel.cpp`) |
| Parallelism | Multi‑threading is used for search and self‑play. | Root‑level Negamax parallelism (`src/core/MoveStrategy.cpp`); self‑play workers in `selfplay/mlp/RLTrainer.cpp` |

## Classes and Structures
| Class/Structure | Relationship |
| --- | --- |
| Player | Base class for HumanPlayer, AIPlayer, HybridPlayer. |
| IMoveStrategy | Interface for RandomStrategy, MonteCarloStrategy, NegamaxStrategy. |
| NegamaxStrategy | Base class for NegamaxHeuristicStrategy and NegamaxGnnStrategy. |
| ValueMLPImpl | Inherits `torch::nn::Module` for LibTorch integration. |
