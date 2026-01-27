# Object-Oriented Design (POO)

The project applies OOP to separate responsibilities: game rules and state live in core classes, move selection is abstracted behind strategy interfaces, and specialized AI strategies inherit from a common Negamax base. This allows the engine to swap evaluation sources (heuristic, MLP, or GNN) without changing the search logic. The implementation also uses RAII and standard library containers for safe resource management, and thread synchronization primitives for parallel search and self‑play data generation.

## Classes and Structures
| Class/Structure | Relationship |
| --- | --- |
| Player | Base class for HumanPlayer, AIPlayer, HybridPlayer. |
| IMoveStrategy | Interface for RandomStrategy, MonteCarloStrategy, NegamaxStrategy. |
| NegamaxStrategy | Base for NegamaxHeuristicStrategy and NegamaxGnnStrategy. |
| ValueMLPImpl | Inherits `torch::nn::Module` for LibTorch integration. |
