#pragma once

#include <string>

#include "gnn/FeatureExtractor.hpp"

// Wrapper for TorchScript model evaluation (value head).
class GNNModel {
public:
    explicit GNNModel(const std::string& modelPath);
    ~GNNModel();

    bool isLoaded() const;
    bool usesCuda() const;

    // Value from player 1 perspective in [-1,1]; flips sign if toMove == 2.
    float evaluate(const FeatureBatch& batch, int toMove) const;

private:
    bool loaded{false};

    struct Impl;
    Impl* impl{nullptr};
};
