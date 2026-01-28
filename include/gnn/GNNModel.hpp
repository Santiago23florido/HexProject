#pragma once

#include <array>
#include <memory>
#include <string>

#include "gnn/FeatureExtractor.hpp"

/**
 * TorchScript wrapper for value model evaluation.
 *
 * Owns a shared implementation and supports optional CUDA execution.
 */
class GNNModel {
public:
    /// Loads a TorchScript model from modelPath.
    explicit GNNModel(const std::string& modelPath, bool preferCuda = false);
    /// Destroys the model wrapper.
    ~GNNModel();

    /// Returns true if a model was loaded successfully.
    bool isLoaded() const;
    /// Returns true if the model is using CUDA.
    bool usesCuda() const;
    /// Returns true if the model expects edge_index in forward.
    bool expectsEdgeIndex() const;

    // Value for the current player in [-1,1] with no sign flip.
    /// Evaluates a graph batch for the given player.
    float evaluate(const FeatureBatch& batch, int toMove) const;
    /// Evaluates a flat feature vector (MLP-style).
    float evaluateFeatures(const std::array<float, 7>& features) const;

private:
    bool loaded{false};
    mutable bool evalLogged{false};

    struct Impl;
    std::shared_ptr<Impl> impl;
};
