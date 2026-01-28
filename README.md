# Hex Project

Hex is a two-player abstract strategy game with perfect information and no chance. This project implements a full C++ Hex system with an object-oriented game engine (board, state, legal moves, BFS win checks), AI search (Negamax with transposition tables), and two evaluation paths: a handcrafted heuristic or a TorchScript value model (MLP by default, optional GNN). A SFML GUI is included for interactive play.

Installation guide: [docs/pages/installation.md](docs/pages/installation.md)

## Windows prerequisites (PowerShell equivalents)
### CMake (download the MSI first)
```powershell
msiexec /i cmake-<version>-windows-x86_64.msi /qn
```

### C++ compiler (MSVC Build Tools)
```powershell
# Run from the folder that contains vs_BuildTools.exe
.\vs_BuildTools.exe --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
```

### LibTorch (download URL from the PyTorch selector)
```powershell
$libtorchUrl = "<URL from PyTorch selector>"
Invoke-WebRequest $libtorchUrl -OutFile "$env:TEMP\libtorch.zip"
Expand-Archive "$env:TEMP\libtorch.zip" -DestinationPath "C:\libtorch" -Force
setx CMAKE_PREFIX_PATH "C:\libtorch\libtorch"
```

### SFML (via vcpkg)
```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
C:\vcpkg\vcpkg install sfml
```

### CUDA Toolkit (optional)
```powershell
# Run the NVIDIA installer you downloaded
cuda_<version>_windows.exe -s
```

### Python 3
```powershell
# Run the Python installer you downloaded
.\python-3.x.x-amd64.exe /passive InstallAllUsers=1 PrependPath=1 Include_launcher=1
```

### Poetry
```powershell
python -m pip install --user poetry
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
rm -rf build
cmake -S . -B build
cmake --build build
cd build
./hex_ui
```
- Builds the main game executable and launches the SFML GUI.


### Self-play training (MLP value model)
```bash
rm -rf selfplay/build
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
