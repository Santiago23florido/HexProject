#pragma once

#include <array>
#include <memory>
#include <string>

#include "gnn/FeatureExtractor.hpp"

// Wrapper for TorchScript model evaluation (value head).
class GNNModel {
public:
    explicit GNNModel(const std::string& modelPath, bool preferCuda = false);
    ~GNNModel();

    bool isLoaded() const;
    bool usesCuda() const;
    bool expectsEdgeIndex() const;

    // Value for the current player in [-1,1] with no sign flip.
    float evaluate(const FeatureBatch& batch, int toMove) const;
    float evaluateFeatures(const std::array<float, 7>& features) const;

private:
    bool loaded{false};
    mutable bool evalLogged{false};

    struct Impl;
    std::shared_ptr<Impl> impl;
};
