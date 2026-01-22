#pragma once

#include <string>

#include <torch/torch.h>

class ValueMLPImpl : public torch::nn::Module {
public:
    ValueMLPImpl(int inputDim = 7, int hidden = 128, int depth = 2, float valueScale = 10000.0f);

    torch::Tensor forward(torch::Tensor x);
    void setNormalization(const torch::Tensor& mean, const torch::Tensor& std);

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

bool saveValueMLPTorchScript(const ValueMLP& model, const std::string& path, std::string* error);
