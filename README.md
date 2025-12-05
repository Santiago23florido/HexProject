# Hex Game Engine (C++)

This project provides an object-oriented implementation of the board game Hex. The engine models the board, game state, coordinate systems, and winner detection in a clear and extensible architecture. It serves as a foundation for future AI methods such as Monte Carlo Tree Search and neural network evaluation (AlphaZero style).

## Game Overview

Hex is a deterministic two-player connection game played on an N×N rhombus-shaped board of hexagonal cells.

- Player X aims to connect the left and right sides of the board.
- Player O aims to connect the top and bottom sides.
- Draws are theoretically impossible; one player must win.

This project supports human vs human and human vs AI (random or Monte Carlo) through a text-based interface.

## Object-Oriented Structure

| Class | Description |
|------|-------------|
| `Board` | Maintains the board matrix, handles move placement, and renders the board to the console. |
| `GameState` | Wraps the board as a logical state, tracks the current player, generates legal moves, and determines if the game is finished. |
| `Cube` | Represents positions using cube coordinates, enabling robust neighbor calculations for winner detection. |
| `Player` hierarchy | `HumanPlayer` reads input, `AIPlayer` delegates to a move strategy. |
| `MoveStrategy` | Interface for AI policies. Includes `RandomStrategy` and `MonteCarloStrategy` (configurable rollouts). |

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

## AI Play

- Default setup: Player X is human, Player O uses `MonteCarloStrategy` (simulation count set in `src/main.cpp`).
- The Monte Carlo bot prints a trace like `[MonteCarlo] move (r,c) wins X/N` on its turn.
- RNG is seeded in `main` with `std::srand(std::time(nullptr))` to avoid repeating sequences.

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
./hex
```

## Usage

- Run `./hex` and follow the prompts. Input moves as `row column` (0-based).
- Board size defaults to 7 (see `Board` constructor in `src/main.cpp`); change it there if needed.
- To change AI strength, adjust the simulation count when constructing `MonteCarloStrategy` in `src/main.cpp`.
- For a quick random bot, construct `AIPlayer(int id)` without passing a strategy.
- For human vs human, instantiate bot