#include "ValueMLP.hpp"

#include <sstream>

#include <torch/script.h>

ValueMLPImpl::ValueMLPImpl(int inputDim, int hidden, int depth, float valueScaleIn)
    : inputDim_(inputDim),
      hidden_(hidden) {
    if (depth < 2) depth = 2;
    if (depth > 3) depth = 3;
    depth_ = depth;

    fc1 = register_module("fc1", torch::nn::Linear(inputDim_, hidden_));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_, hidden_));
    if (depth_ == 3) {
        fc3 = register_module("fc3", torch::nn::Linear(hidden_, hidden_));
    }
    out = register_module("out", torch::nn::Linear(hidden_, 1));

    mean = register_buffer("mean", torch::zeros({inputDim_}));
    std = register_buffer("std", torch::ones({inputDim_}));
    valueScale = register_buffer("value_scale", torch::tensor({valueScaleIn}));
}

torch::Tensor ValueMLPImpl::forward(torch::Tensor x) {
    x = (x - mean) / std;
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    if (depth_ == 3 && fc3) {
        x = torch::relu(fc3->forward(x));
    }
    x = out->forward(x).squeeze(1);
    return x * valueScale;
}

void ValueMLPImpl::setNormalization(const torch::Tensor& meanIn, const torch::Tensor& stdIn) {
    const auto meanDev = meanIn.to(mean.device());
    const auto stdDev = stdIn.to(std.device()).clamp_min(1e-6);
    mean.copy_(meanDev);
    std.copy_(stdDev);
}

static std::string buildForwardSource(int depth) {
    std::ostringstream ss;
    ss << "def forward(self, x):\n";
    ss << "    x = (x - self.mean) / self.std\n";
    ss << "    x = torch.matmul(x, self.fc1_weight.t()) + self.fc1_bias\n";
    ss << "    x = torch.relu(x)\n";
    ss << "    x = torch.matmul(x, self.fc2_weight.t()) + self.fc2_bias\n";
    ss << "    x = torch.relu(x)\n";
    if (depth == 3) {
        ss << "    x = torch.matmul(x, self.fc3_weight.t()) + self.fc3_bias\n";
        ss << "    x = torch.relu(x)\n";
    }
    ss << "    x = torch.matmul(x, self.out_weight.t()) + self.out_bias\n";
    ss << "    x = x.squeeze(1)\n";
    ss << "    return x * self.value_scale\n";
    return ss.str();
}

bool saveValueMLPTorchScript(const ValueMLP& model, const std::string& path, std::string* error) {
    try {
        torch::jit::Module module("ValueMLP");

        auto fc1Weight = model->fc1->weight.detach().cpu();
        auto fc1Bias = model->fc1->bias.detach().cpu();
        auto fc2Weight = model->fc2->weight.detach().cpu();
        auto fc2Bias = model->fc2->bias.detach().cpu();
        auto outWeight = model->out->weight.detach().cpu();
        auto outBias = model->out->bias.detach().cpu();

        module.register_parameter("fc1_weight", fc1Weight, false);
        module.register_parameter("fc1_bias", fc1Bias, false);
        module.register_parameter("fc2_weight", fc2Weight, false);
        module.register_parameter("fc2_bias", fc2Bias, false);
        if (model->depth() == 3) {
            auto fc3Weight = model->fc3->weight.detach().cpu();
            auto fc3Bias = model->fc3->bias.detach().cpu();
            module.register_parameter("fc3_weight", fc3Weight, false);
            module.register_parameter("fc3_bias", fc3Bias, false);
        }
        module.register_parameter("out_weight", outWeight, false);
        module.register_parameter("out_bias", outBias, false);

        module.register_buffer("mean", model->mean.detach().cpu());
        module.register_buffer("std", model->std.detach().cpu());
        module.register_buffer("value_scale", model->valueScale.detach().cpu());

        module.define(buildForwardSource(model->depth()));
        module.save(path);
        return true;
    } catch (const std::exception& e) {
        if (error) {
            *error = e.what();
        }
        return false;
    }
}
