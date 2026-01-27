#pragma once

#include <string>

#include <torch/torch.h>

/**
 * Small MLP for value estimation with input normalization and scaling.
 *
 * Owns parameters and buffers; output is multiplied by valueScale.
 */
class ValueMLPImpl : public torch::nn::Module {
public:
    /// Creates a value MLP with the given dimensions and output scale.
    ValueMLPImpl(int inputDim = 7, int hidden = 128, int depth = 2, float valueScale = 10000.0f);

    /// Evaluates the network on x and returns scaled outputs.
    torch::Tensor forward(torch::Tensor x);
    /// Sets mean/std normalization buffers used in forward.
    void setNormalization(const torch::Tensor& mean, const torch::Tensor& std);

    /// Returns the number of hidden layers.
    int depth() const { return depth_; }

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
    torch::nn::Linear out{nullptr};

    torch::Tensor mean;
    torch::Tensor std;
    torch::Tensor valueScale;

private:
    int inputDim_{7};
    int hidden_{128};
    int depth_{2};
};

TORCH_MODULE(ValueMLP);

/// Exports a ValueMLP to TorchScript at path; returns false and sets error on failure.
bool saveValueMLPTorchScript(const ValueMLP& model, const std::string& path, std::string* error);
