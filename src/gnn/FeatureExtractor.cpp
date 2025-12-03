#include "gnn/FeatureExtractor.hpp"

#include <algorithm>
#include <cmath>

#include "gnn/Graph.hpp"

Graph& FeatureExtractor::getGraph(int N) {
    auto it = cache.find(N);
    if (it != cache.end()) {
        return it->second;
    }
    Graph g = buildHexGraph(N, true);
    auto inserted = cache.emplace(N, std::move(g));
    return inserted.first->second;
}

void FeatureExtractor::flatten(const Graph& g, FeatureBatch& batch) const {
    batch.N = g.N;
    batch.numNodes = g.numNodes;
    batch.featureDim = FEATURE_DIM;
    batch.nodeFeatures.clear();
    batch.nodeFeatures.reserve(static_cast<size_t>(g.numNodes) * FEATURE_DIM);

    for (int i = 0; i < g.numNodes; ++i) {
        const NodeFeatures& nf = g.features[i];
        batch.nodeFeatures.push_back(nf.p1);
        batch.nodeFeatures.push_back(nf.p2);
        batch.nodeFeatures.push_back(nf.empty);
        batch.nodeFeatures.push_back(nf.sideA);
        batch.nodeFeatures.push_back(nf.sideB);
        batch.nodeFeatures.push_back(nf.degree);
        batch.nodeFeatures.push_back(nf.distToA);
        batch.nodeFeatures.push_back(nf.distToB);
    }

    batch.edgeSrc.clear();
    batch.edgeDst.clear();
    for (int src = 0; src < g.numNodes; ++src) {
        for (int dst : g.adj[src]) {
            batch.edgeSrc.push_back(src);
            batch.edgeDst.push_back(dst);
        }
    }
}

FeatureBatch FeatureExtractor::toBatch(const Board& board) {
    const int N = board.N;
    Graph& g = getGraph(N);
    fillFeatures(g, board);

    FeatureBatch batch;
    flatten(g, batch);
    return batch;
}

FeatureBatch FeatureExtractor::toBatch(const GameState& state) {
    const auto linear = state.LinearBoard();
    const int N = static_cast<int>(std::sqrt(linear.size()));
    Graph& g = getGraph(N);
    fillFeatures(g, state);

    FeatureBatch batch;
    flatten(g, batch);
    return batch;
}
