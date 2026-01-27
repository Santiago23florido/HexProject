# Introduction

Hex is a two-player abstract strategy game with perfect information and no chance. Players alternate placing stones on a hexagonal grid, each trying to connect their assigned opposite sides first. This project implements a 7×7 board by default (configurable), supports human-vs-human or human-vs-AI play, and provides AI opponents based on Negamax search with either a handcrafted heuristic or a learned MLP value model trained from self-play.

The report also highlights how object‑oriented design is used to structure the engine, AI strategies, and supporting utilities.

## Classes and Structures
| Class/Structure | Role |
| --- | --- |
| Board (class) | Board representation and stone placement rules. |
| GameState (class) | Turn tracking and winner detection. |
| Player (class) | Player identity and move policy wiring. |
| IMoveStrategy (class) | Strategy interface for selecting moves. |
| NegamaxStrategy (class) | Search-based AI used by the agents. |
