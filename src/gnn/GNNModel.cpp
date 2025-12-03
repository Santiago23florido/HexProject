#include "gnn/GNNModel.hpp"

#include <iostream>
#include <vector>

#include <torch/script.h>

struct GNNModel::Impl {
    torch::jit::script::Module module;
};

GNNModel::GNNModel(const std::string& modelPath) {
    try {
        impl = new Impl();
        impl->module = torch::jit::load(modelPath);
        impl->module.eval();
        loaded = true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load TorchScript model: " << e.what() << "\n";
        loaded = false;
        delete impl;
        impl = nullptr;
    }
}

bool GNNModel::isLoaded() const {
    return loaded;
}

float GNNModel::evaluate(const FeatureBatch& batch, int toMove) const {
    if (!loaded || impl == nullptr) return 0.0f;

    torch::NoGradGuard no_grad;
    torch::Tensor x = torch::from_blob(const_cast<float*>(batch.nodeFeatures.data()),
                                       {batch.numNodes, batch.featureDim},
                                       torch::TensorOptions().dtype(torch::kFloat32));

    // Edge index: two rows, E columns, owning its memory
    torch::Tensor src = torch::tensor(batch.edgeSrc, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor dst = torch::tensor(batch.edgeDst, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor edge_index = torch::stack({src, dst}, 0);

    auto output = impl->module.forward({x, edge_index}).toTensor().item<float>();
    float val = output;
    if (toMove == 2) val = -val;
    return val;
}

GNNModel::~GNNModel() {
    delete impl;
}
