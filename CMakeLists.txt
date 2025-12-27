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

# ==== Find SFML ====
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# ==== Core library (engine + strategies + GNN) ====
add_library(hex_core
    src/core/Board.cpp
    src/core/GameState.cpp
    src/core/Cube.cpp
    src/core/Player.cpp
    src/core/MoveStrategy.cpp
    src/gnn/Graph.cpp
    src/gnn/FeatureExtractor.cpp
    src/gnn/GNNModel.cpp
)

target_include_directories(hex_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/core
    ${CMAKE_CURRENT_SOURCE_DIR}/include/gnn
)

# ==== Link with libtorch ====
target_link_libraries(hex_core PUBLIC "${TORCH_LIBRARIES}")

# Define USE_TORCH so your #ifdef USE_TORCH code is enabled
target_compile_definitions(hex_core PUBLIC USE_TORCH)

# ==== Executables ====
add_executable(hex src/cli/main.cpp)
target_link_libraries(hex PRIVATE hex_core)

add_executable(hex_ui
    src/ui/main.cpp
    src/ui/HexGameUI.cpp
    src/ui/ImageViewer.cpp
    src/ui/HexTile.cpp
)
target_include_directories(hex_ui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(hex_ui PRIVATE
    hex_core
    sfml-graphics
    sfml-window
    sfml-system
)

# ==== RPATH so the binaries can find libtorch .so files at runtime ====
foreach(target_name IN ITEMS hex hex_ui)
    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "$ENV{HOME}/libtorch/lib"
        INSTALL_RPATH "$ENV{HOME}/libtorch/lib"
    )
endforeach()
