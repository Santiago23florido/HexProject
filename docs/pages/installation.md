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

## LibTorch installation (details)
- Download the LibTorch C++ distribution from the official PyTorch site.
- Pick the correct ABI:
  - Linux GCC/Clang: use the **cxx11 ABI** build.
  - Windows: select the MSVC build that matches your compiler.
- Choose CPU or CUDA based on your environment.
- Extract to a known location (e.g., `~/libtorch`).

### Download (CPU by default)
```bash
wget https://download.pytorch.org/libtorch/\
nightly/cpu/libtorch-shared-with-deps-latest.zip \
  -O libtorch-cpu.zip
unzip libtorch-cpu.zip
```
For CUDA, download a build that matches your toolkit version from the PyTorch selector.

### Configure CMake
```bash
export Torch_DIR="$HOME/libtorch/share/cmake/Torch"

```

### Runtime loader path (if needed)
```bash
export LD_LIBRARY_PATH="$HOME/libtorch/lib:$LD_LIBRARY_PATH"
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

## If a prerequisite is missing
- CMake: install via your OS package manager or official binaries.
  - Debian/Ubuntu: `sudo apt-get install cmake`
- C++ compiler: install GCC/Clang (or MSVC on Windows) with C++17 support.
  - Debian/Ubuntu: `sudo apt-get install g++` or `sudo apt-get install clang`
- LibTorch: download the C++ distribution for your OS/ABI; set `Torch_DIR` or `CMAKE_PREFIX_PATH`.
- SFML: install development packages (e.g., `libsfml-dev`).
  - Debian/Ubuntu: `sudo apt-get install libsfml-dev`
- CUDA (optional): install a toolkit that matches your LibTorch build; otherwise use a CPU-only LibTorch build.
- Python + poetry (optional): install Python 3 and poetry.
  - Debian/Ubuntu: `sudo apt-get install python3 python3-pip`
  - Poetry: `python3 -m pip install --user poetry`

## Version compatibility
- CMake: 3.18+ (required by the top-level CMakeLists).
- C++ compiler: full C++17 support (e.g., GCC 7+, Clang 6+, MSVC 2019+).
- SFML: 2.5+ (CMake requests 2.5).
- LibTorch: C++ distribution matching your OS and compiler ABI; if using CUDA, match the CUDA version in the LibTorch build.
- CUDA: must match the LibTorch CUDA build; otherwise use the CPU build.
If a version is incompatible, upgrade/downgrade to the minimums above or pick matching LibTorch/CUDA binaries.

## Linux: add to PATH (if needed)
```bash
# ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"     # poetry
export PATH="/opt/cmake/bin:$PATH"       # cmake (archive)
export PATH="/usr/local/cuda/bin:$PATH"  # cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:\
$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/libtorch/lib:\
$LD_LIBRARY_PATH"
source ~/.bashrc
```

## Build and run the game
```bash
rm -rf build
cmake -S . -B build
cmake --build build
cd build
./hex_ui
```

## Build and run self-play
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

## Fast-installation (Linux, gameplay only)
### Download
```bash
gdown --fuzzy \
  "https://drive.google.com/file/d/1iVbLBNMKzCeTfzk1X_cW3bPPNIjGCMX-/view?usp=sharing" \
  -O HexProject-CPU-x86_64.AppImage
```

### Verify
```bash
file HexProject-CPU-x86_64.AppImage
```

### Run
```bash
chmod +x HexProject-CPU-x86_64.AppImage
./HexProject-CPU-x86_64.AppImage
```
