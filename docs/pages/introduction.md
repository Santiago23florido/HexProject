# Game Rules and Overview

Hex is a two-player abstract strategy game with perfect information and no chance. Players alternate placing stones on a hexagonal grid; each player owns a pair of opposite sides of the board. The goal is to be the first to form a connected chain of your stones linking your two sides.

Key rules (as used in this project):
- The board is hexagonal in adjacency (implemented on a 7×7 grid by default, configurable).
- Players take turns placing one stone in an empty cell.
- There are no draws: a winner always exists because any fully played Hex board contains a connecting path for one player.
- Win detection is performed by exploring connected components across the board’s opposing sides.

This project also includes AI opponents based on Negamax search, using either a handcrafted heuristic or a learned value model trained via self-play.
