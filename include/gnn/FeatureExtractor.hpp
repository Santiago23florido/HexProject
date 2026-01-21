#pragma once

#include <unordered_map>
#include <vector>

#include "core/Board.hpp"
#include "core/GameState.hpp"
#include "gnn/Graph.hpp"

// Flattened tensors ready for a GNN backend
struct FeatureBatch {
    int N{0};
    int numNodes{0};
    int featureDim{0};
    std::vector<float> nodeFeatures; // shape: numNodes * featureDim, row-major
    std::vector<int> edgeSrc;        // edge list source indices
    std::vector<int> edgeDst;        // edge list destination indices
};

class FeatureExtractor {
public:
    FeatureExtractor() = default;

    // Build or reuse the graph for a given size, then fill features from Board
    FeatureBatch toBatch(const Board& board) const;
    void toBatch(const Board& board, FeatureBatch& batch) const;
    // Overload for GameState convenience
    FeatureBatch toBatch(const GameState& state) const;
    void toBatch(const GameState& state, FeatureBatch& batch) const;

private:
    Graph& getGraph(int N) const;
    void flatten(const Graph& g, FeatureBatch& batch) const;

    mutable std::unordered_map<int, Graph> cache; // graphs by board size
    static constexpr int FEATURE_DIM = 10; // p1, p2, empty, sideA, sideB, degree, distToA, distToB, toMoveP1, toMoveP2
};
