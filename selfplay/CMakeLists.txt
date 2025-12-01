add_executable(selfplay
    selfplay/main.cpp
    selfplay/GameRunner.cpp
    selfplay/DataCollector.cpp
    selfplay/Serializer.cpp
)

target_include_directories(selfplay PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/gnn
)

target_compile_features(selfplay PRIVATE cxx_std_17)
