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
)

target_include_directories(hex PUBLIC include)
