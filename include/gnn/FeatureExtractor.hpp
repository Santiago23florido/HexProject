#pragma once

#include <unordered_map>
#include <vector>

#include "core/Board.hpp"
#include "core/GameState.hpp"
#include "gnn/Graph.hpp"

/**
 * Flattened tensors ready for a GNN backend.
 */
struct FeatureBatch {
    int N{0};
    int numNodes{0};
    int featureDim{0};
    std::vector<float> nodeFeatures; // shape: numNodes * featureDim, row-major
    std::vector<int> edgeSrc;        // edge list source indices
    std::vector<int> edgeDst;        // edge list destination indices
};

/**
 * Extracts graph features and packs them into batches.
 *
 * Caches one graph per board size.
 */
class FeatureExtractor {
public:
    FeatureExtractor() = default;

    /// Builds or reuses the graph for a board and returns a batch.
    FeatureBatch toBatch(const Board& board) const;
    /// Fills an existing batch from a board.
    void toBatch(const Board& board, FeatureBatch& batch) const;
    /// Builds or reuses the graph for a state and returns a batch.
    FeatureBatch toBatch(const GameState& state) const;
    /// Fills an existing batch from a state.
    void toBatch(const GameState& state, FeatureBatch& batch) const;

private:
    Graph& getGraph(int N) const;
    void flatten(const Graph& g, FeatureBatch& batch) const;

    mutable std::unordered_map<int, Graph> cache; // graphs by board size
    static constexpr int FEATURE_DIM = 10; // p1, p2, empty, sideA, sideB, degree, distToA, distToB, toMoveP1, toMoveP2
};
