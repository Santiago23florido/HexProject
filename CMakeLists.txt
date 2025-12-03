cmake_minimum_required(VERSION 3.18)

project(HexProject LANGUAGES CXX)

# ==== C++ standard ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==== Default path for libtorch (Torch_DIR) ====
# We assume you extracted libtorch into: $HOME/libtorch
if(NOT Torch_DIR)
    set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake")
endif()

# Optional but useful: let CMake also search using CMAKE_PREFIX_PATH
if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
endif()

# ==== CUDA configuration for libtorch with GPU (cu128) ====
# We do NOT set CMAKE_CUDA_ARCHITECTURES to avoid PyTorch warnings.
# Instead, tell PyTorch explicitly which GPU arch to target.
# RTX 4070 = Ada, SM 89 → arch "8.9"
set(ENV{TORCH_CUDA_ARCH_LIST} "8.9")

# Make sure CMake knows where nvcc is
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

# ==== Find Torch (libtorch) ====
find_package(Torch REQUIRED)

# ==== Main executable ====
add_executable(hex
    src/main.cpp
    src/Board.cpp
    src/GameState.cpp
    src/Cube.cpp
    src/Player.cpp
    src/MoveStrategy.cpp
    src/gnn/Graph.cpp
    src/gnn/FeatureExtractor.cpp
    src/gnn/GNNModel.cpp
)

target_include_directories(hex PUBLIC
    include
    include/gnn
)

# ==== Link with libtorch ====
target_link_libraries(hex PRIVATE "${TORCH_LIBRARIES}")

# Define USE_TORCH so your #ifdef USE_TORCH code is enabled
target_compile_definitions(hex PRIVATE USE_TORCH)

# ==== RPATH so the binary can find libtorch .so files at runtime ====
set_target_properties(hex PROPERTIES
    BUILD_RPATH "$ENV{HOME}/libtorch/lib"
    INSTALL_RPATH "$ENV{HOME}/libtorch/lib"
)
