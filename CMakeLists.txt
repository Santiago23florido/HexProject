cmake_minimum_required(VERSION 3.10)
project(HexProject)

set(CMAKE_CXX_STANDARD 17)

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

target_include_directories(hex PUBLIC include include/gnn)

find_package(Torch REQUIRED)
target_link_libraries(hex PRIVATE ${TORCH_LIBRARIES})
