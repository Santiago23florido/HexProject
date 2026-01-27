# Core Module

The core module encapsulates the game state, players, and decision strategies. It defines the Hex board, state updates, winner detection, player abstractions, and move‑selection strategies (including Negamax variants). It also includes utilities such as Zobrist hashing for transposition tables and feature computation used by the heuristics.

## Classes and Structures
| Class/Structure (type) | Direct derivatives | Description |
| --- | --- | --- |
| Board (class) |  | Board representation and stone placement. |
| GameState (class) |  | Game state updates and winner detection. |
| Cube (class) |  | Cube coordinates for hex‑grid adjacency. |
| Player (class) | HumanPlayer, AIPlayer, HybridPlayer | Base player abstraction. |
| IMoveStrategy (class) | RandomStrategy, MonteCarloStrategy, NegamaxStrategy | Strategy interface for move selection. |
| NegamaxStrategy (class) | NegamaxHeuristicStrategy, NegamaxGnnStrategy | Negamax search with evaluation. |
| Zobrist (class) |  | Hashing for transposition tables. |
