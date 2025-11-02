# Hex Game Engine (C++)

This project implements the board game **Hex**, including:
- Board representation
- Coordinate conversion to cube coordinates
- Winner detection using BFS on cube neighbors
- Text-based human-vs-human gameplay

The goal is to later integrate MCTS + Neural Network (AlphaZero style).

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
./hex
