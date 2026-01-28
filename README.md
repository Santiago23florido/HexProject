# Hex Project

Hex is a two-player abstract strategy game with perfect information and no chance. This project implements a full C++ Hex system with an object-oriented game engine (board, state, legal moves, BFS win checks), AI search (Negamax with transposition tables), and two evaluation paths: a handcrafted heuristic or a TorchScript value model (MLP by default, optional GNN). A SFML GUI is included for interactive play.

Installation guide: [docs/pages/installation.md](docs/pages/installation.md)

## Fast-play (no build)
### WSL (Ubuntu)
Requires WSL with Ubuntu 24.04.3 LTS or newer.
Check with `lsb_release -a`.

#### Download
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install gdown
gdown --fuzzy \
  "https://drive.google.com/file/d/1iVbLBNMKzCeTfzk1X_cW3bPPNIjGCMX-/view?usp=sharing" \
  -O HexProject-CPU-x86_64.AppImage
```

#### Verify
```bash
file HexProject-CPU-x86_64.AppImage
```

#### Run
```bash
chmod +x HexProject-CPU-x86_64.AppImage
./HexProject-CPU-x86_64.AppImage
```

### Windows (gameplay only)
Open the link, download, and run the installer:
```
https://drive.google.com/file/d/1XExLIMiUIn2Q0FxtC6UjaX-f9eCW9wgK/view?usp=drive_link
```

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
cd build
./hex_ui
```
- Builds the main game executable and launches the SFML GUI.


### Self-play training (MLP value model)
```bash
cmake -S selfplay -B selfplay/build
cmake --build selfplay/build

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
