#pragma once

#include <unordered_map>
#include <vector>

#include "Board.hpp"
#include "GameState.hpp"
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
    FeatureBatch toBatch(const Board& board);
    // Overload for GameState convenience
    FeatureBatch toBatch(const GameState& state);

private:
    Graph& getGraph(int N);
    void flatten(const Graph& g, FeatureBatch& batch) const;

    std::unordered_map<int, Graph> cache; // graphs by board size
    static constexpr int FEATURE_DIM = 6; // p1, p2, empty, sideA, sideB, degree
};
