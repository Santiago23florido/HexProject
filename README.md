# Hex Project

Hex is a two-player abstract strategy game with perfect information and no chance. This project implements a full C++ Hex system with an object-oriented game engine (board, state, legal moves, BFS win checks), AI search (Negamax with transposition tables), and two evaluation paths: a handcrafted heuristic or a TorchScript value model (MLP by default, optional GNN). A SFML GUI is included for interactive play.

Installation guide: [docs/pages/installation.md](docs/pages/installation.md)

Documentation map:
- Architecture overview: [docs/pages/architecture_overview.md](docs/pages/architecture_overview.md)
- Introduction: [docs/pages/introduction.md](docs/pages/introduction.md)
- Project structure: [docs/pages/project_structure.md](docs/pages/project_structure.md)
- Core module: [docs/pages/core.md](docs/pages/core.md)
- MLP/GNN module: [docs/pages/gnn.md](docs/pages/gnn.md)
- Self-play module: [docs/pages/selfplay.md](docs/pages/selfplay.md)
- UI module: [docs/pages/ui.md](docs/pages/ui.md)
- OOP design notes: [docs/pages/oop.md](docs/pages/oop.md)
- Installation: [docs/pages/installation.md](docs/pages/installation.md)

## Build and Run

### Game UI
```bash
cmake -S . -B build
cmake --build build
./build/hex_ui
```
- Builds the main game executable and launches the SFML GUI.


### Self-play training (MLP value model)
```bash
./selfplay/build/selfplay \
  --selfplay-train \
  --export-ts \
  --train-games 200 \
  --min-depth 10 \
  --max-depth 20 \
  --batch-size 256 \
  --updates-per-game 1 \
  --device cuda
```
- Trains the ValueMLP from self-play data and optionally exports TorchScript to `scripts/models/`.
