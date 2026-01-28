cmake_minimum_required(VERSION 3.18)

project(HexProject LANGUAGES CXX)

# ==== C++ standard ====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==== Default path for libtorch (Torch_DIR) ====
if(NOT Torch_DIR)
    set(Torch_DIR "$ENV{HOME}/libtorch/share/cmake/Torch" CACHE PATH "Path to TorchConfig.cmake")
endif()


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

set(HEX_HAS_CUDA OFF)
if(CUDAToolkit_FOUND)
    set(TORCH_HAS_CUDA OFF)
    if(DEFINED TORCH_LIBRARY)
        get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
    elseif(DEFINED Torch_DIR)
        get_filename_component(TORCH_PREFIX "${Torch_DIR}/../../.." ABSOLUTE)
        set(TORCH_LIB_DIR "${TORCH_PREFIX}/lib")
    endif()
    if(TORCH_LIB_DIR)
        if(WIN32)
            set(_torch_cuda_libs "c10_cuda.dll;torch_cuda.dll")
        elseif(APPLE)
            set(_torch_cuda_libs "libc10_cuda.dylib;libtorch_cuda.dylib")
        else()
            set(_torch_cuda_libs "libc10_cuda.so;libtorch_cuda.so")
        endif()
        foreach(_lib ${_torch_cuda_libs})
            if(EXISTS "${TORCH_LIB_DIR}/${_lib}")
                set(TORCH_HAS_CUDA ON)
            endif()
        endforeach()
    endif()
    if(TORCH_HAS_CUDA)
        set(HEX_HAS_CUDA ON)
    endif()
endif()

# ==== Find SFML ====
find_package(SFML 2.5 COMPONENTS graphics window system audio REQUIRED)

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
target_include_directories(hex_core SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})

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
if(WIN32)
    set_target_properties(hex_ui PROPERTIES WIN32_EXECUTABLE ON)
endif()
target_link_libraries(hex_ui PRIVATE
    hex_core
    sfml-graphics
    sfml-window
    sfml-system
    sfml-audio
)
if(WIN32)
    target_link_libraries(hex_ui PRIVATE sfml-main)
endif()
if(HEX_HAS_CUDA)
    target_compile_definitions(hex_ui PRIVATE HEX_HAS_CUDA)
    target_link_libraries(hex_ui PRIVATE CUDA::cudart)
endif()

# ==== RPATH so the binaries can find libtorch .so files at runtime ====
foreach(target_name IN ITEMS hex hex_ui)
    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "$ENV{HOME}/libtorch/lib"
        INSTALL_RPATH "$ENV{HOME}/libtorch/lib"
    )
endforeach()
