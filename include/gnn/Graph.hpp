#pragma once
#include "core/Board.hpp"
#include "core/GameState.hpp"
#include <vector>


using NodeId = int;


/**
 * Per-node feature bundle for Hex graphs.
 */
struct NodeFeatures {
    float p1{0.f};     // 1.0 if occupied by player 1
    float p2{0.f};     // 1.0 if occupied by player 2
    float empty{0.f};  // 1.0 if the cell is empty
    float sideA{0.f};  // 1.0 if cell touches player 1 target sides (columns 0 or N-1)
    float sideB{0.f};  // 1.0 if cell touches player 2 target sides (rows 0 or N-1)
    float degree{0.f}; // normalized degree (neighbors / 6.0)
    float distToA{0.f}; // normalized shortest hops to supernode A (columns), 0 if not used
    float distToB{0.f}; // normalized shortest hops to supernode B (rows), 0 if not used
    float toMoveP1{0.f}; // one-hot current player (filled at inference)
    float toMoveP2{0.f};
};


/**
 *  Graph representation of a Hex board with optional super-nodes.
 *
 * Owns adjacency and feature arrays; N is the board side length.
 */
struct Graph {
    int numNodes{0};
    int N{0}; 
    std::vector<std::vector<NodeId>> adj;
    std::vector<NodeFeatures> features;   
    int superOffset{-1};                 
    int superCount{0};                    
};


/// Builds a Hex adjacency graph, optionally adding border super-nodes.
Graph buildHexGraph(int N, bool addBorderSuperNodes = true);


/// Fills dynamic occupancy features from a Board.
void fillFeatures(Graph& g, const Board& board);


/// Fills dynamic occupancy features from a GameState.
void fillFeatures(Graph& g, const GameState& state);
