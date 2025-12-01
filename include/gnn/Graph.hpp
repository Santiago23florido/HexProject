#pragma once
#include "../Board.hpp"
#include "../GameState.hpp"
#include <vector>


using NodeId = int;


struct NodeFeatures {
    float p1{0.f};     // 1.0 if occupied by player 1
    float p2{0.f};     // 1.0 if occupied by player 2
    float empty{0.f};  // 1.0 if the cell is empty
    float sideA{0.f};  // 1.0 if cell touches player 1 target sides (columns 0 or N-1)
    float sideB{0.f};  // 1.0 if cell touches player 2 target sides (rows 0 or N-1)
    float degree{0.f}; // normalized degree (neighbors / 6.0)
};


struct Graph {
    int numNodes{0};
    int N{0}; 
    std::vector<std::vector<NodeId>> adj;
    std::vector<NodeFeatures> features;   
    int superOffset{-1};                 
    int superCount{0};                    
};


Graph buildHexGraph(int N, bool addBorderSuperNodes = true);


void fillFeatures(Graph& g, const Board& board);


void fillFeatures(Graph& g, const GameState& state);
