cmake_minimum_required(VERSION 3.16)
project(SelfPlay LANGUAGES CXX)

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(selfplay
    main.cpp
    GameRunner.cpp
    DataCollector.cpp
    Serializer.cpp
    ${ROOT_DIR}/src/Board.cpp
    ${ROOT_DIR}/src/GameState.cpp
    ${ROOT_DIR}/src/Cube.cpp
    ${ROOT_DIR}/src/Player.cpp
    ${ROOT_DIR}/src/MoveStrategy.cpp
)

target_include_directories(selfplay PRIVATE
    ${ROOT_DIR}/include
    ${ROOT_DIR}/include/gnn
)

target_compile_features(selfplay PRIVATE cxx_std_17)
target_compile_options(selfplay PRIVATE -fPIC)
