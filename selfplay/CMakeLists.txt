cmake_minimum_required(VERSION 3.18)

project(SelfPlay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# ==== Default path for libtorch (Torch_DIR) ====
if(NOT Torch_DIR)
    set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake")
endif()
# Optional but useful: let CMake also search using CMAKE_PREFIX_PATH
if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
endif()

# ==== CUDA configuration for libtorch with GPU (cu128) ====
set(ENV{TORCH_CUDA_ARCH_LIST} "8.9")

# Make sure CMake knows where nvcc is if CUDA is installed
if(NOT DEFINED CMAKE_CUDA_COMPILER AND EXISTS "/usr/local/cuda-12.8/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()

# ==== Find Torch (libtorch) ====
find_package(Torch REQUIRED)
find_package(CUDAToolkit QUIET)

set(SELFPLAY_HAS_CUDA OFF)
if(CUDAToolkit_FOUND)
    set(TORCH_HAS_CUDA OFF)
    if(DEFINED TORCH_CUDA_LIBRARIES AND TORCH_CUDA_LIBRARIES)
        set(TORCH_HAS_CUDA ON)
    elseif(DEFINED TORCH_LIBRARY)
        get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
        if(EXISTS "${TORCH_LIB_DIR}/libc10_cuda.so" OR EXISTS "${TORCH_LIB_DIR}/libtorch_cuda.so")
            set(TORCH_HAS_CUDA ON)
        endif()
    endif()
    if(TORCH_HAS_CUDA)
        set(SELFPLAY_HAS_CUDA ON)
    endif()
endif()

# ==== Selfplay executable ====
add_executable(selfplay
    main.cpp
    gnn/GameRunner.cpp
    gnn/DataCollector.cpp
    gnn/Serializer.cpp
    mlp/ReplayBuffer.cpp
    mlp/ValueMLP.cpp
    mlp/RLTrainer.cpp
    ${ROOT_DIR}/src/core/Board.cpp
    ${ROOT_DIR}/src/core/GameState.cpp
    ${ROOT_DIR}/src/core/Cube.cpp
    ${ROOT_DIR}/src/core/Player.cpp
    ${ROOT_DIR}/src/core/MoveStrategy.cpp
    ${ROOT_DIR}/src/gnn/Graph.cpp
    ${ROOT_DIR}/src/gnn/FeatureExtractor.cpp
    ${ROOT_DIR}/src/gnn/GNNModel.cpp
)

target_include_directories(selfplay PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${ROOT_DIR}/include
    ${ROOT_DIR}/include/core
    ${ROOT_DIR}/include/gnn
)

# ==== Link with libtorch ====
target_link_libraries(selfplay PRIVATE "${TORCH_LIBRARIES}")

# Embed repo root so default model paths stay stable after moving files.
target_compile_definitions(selfplay PRIVATE SELFPLAY_ROOT_DIR="${ROOT_DIR}")

# Define USE_TORCH so #ifdef USE_TORCH code is enabled
target_compile_definitions(selfplay PRIVATE USE_TORCH)
if(SELFPLAY_HAS_CUDA)
    target_compile_definitions(selfplay PRIVATE SELFPLAY_HAS_CUDA)
endif()

# ==== RPATH so the binary can find libtorch .so files at runtime ====
set_target_properties(selfplay PROPERTIES
    BUILD_RPATH "$ENV{HOME}/libtorch/lib"
    INSTALL_RPATH "$ENV{HOME}/libtorch/lib"
)
