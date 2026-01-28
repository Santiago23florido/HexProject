# Installation

## Clone the repository
```bash
git clone https://github.com/Santiago23florido/HexProject.git
cd HexProject
```

## Prerequisites
### CMake
```bash
cmake --version
```

### C++17 compiler
```bash
g++ --version
# or: clang++ --version
```

### LibTorch (TorchScript)
```bash
ls $HOME/libtorch/share/cmake/Torch
# or set Torch_DIR/CMAKE_PREFIX_PATH
```

### SFML 2.5
```bash
pkg-config --modversion sfml-all
# or: sudo apt-get install libsfml-dev
```

### CUDA Toolkit (optional)
```bash
nvcc --version
```

### Python 3 + poetry (optional)
```bash
python3 --version
poetry --version
```

## Build and run the game
```bash
cmake -S . -B build
cmake --build build
cd build
./hex_ui
```

## Build and run self-play
```bash
cmake -S selfplay -B selfplay/build
cmake --build selfplay/build
./selfplay/build/selfplay \
  <games> <minDepth> \
  <maxDepth> <outputPath> \
  <minPairs> <maxPairs> \
  <timeLimitMs>
# or training/export:
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
