#include "gnn/GNNModel.hpp"

#include <filesystem>
#include <iostream>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

struct GNNModel::Impl {
    torch::jit::script::Module module;
    torch::Device device{torch::kCPU};
    bool useCuda{false};
};

GNNModel::GNNModel(const std::string& modelPath) {
    if (modelPath.empty()) {
        return;
    }

    namespace fs = std::filesystem;
    fs::path path(modelPath);
    if (!path.is_absolute()) {
        path = fs::current_path() / path;
    }
    path = path.lexically_normal();
    if (!fs::exists(path)) {
        std::cerr << "[GNN] Model file not found at " << path
                  << " (falling back to heuristic evaluation)\n";
        return;
    }

    try {
        impl = new Impl();
        impl->module = torch::jit::load(path.string());
        impl->useCuda = torch::cuda::is_available();
        impl->device = impl->useCuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
        impl->module.to(impl->device);
        impl->module.eval();
        std::cout << "[GNN] Loaded model from " << path.string()
                  << " | device: " << (impl->useCuda ? "CUDA" : "CPU") << "\n";
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

bool GNNModel::usesCuda() const {
    return loaded && impl != nullptr && impl->useCuda;
}

float GNNModel::evaluate(const FeatureBatch& batch, int toMove) const {
    if (!loaded || impl == nullptr) return 0.0f;

    torch::NoGradGuard no_grad;
    torch::Tensor x = torch::from_blob(const_cast<float*>(batch.nodeFeatures.data()),
                                       {batch.numNodes, batch.featureDim},
                                       torch::TensorOptions().dtype(torch::kFloat32)).to(impl->device);

    // Edge index: two rows, E columns, owning its memory
    torch::Tensor src = torch::tensor(batch.edgeSrc, torch::TensorOptions().dtype(torch::kInt64)).to(impl->device);
    torch::Tensor dst = torch::tensor(batch.edgeDst, torch::TensorOptions().dtype(torch::kInt64)).to(impl->device);
    torch::Tensor edge_index = torch::stack({src, dst}, 0);

    auto output = impl->module.forward({x, edge_index}).toTensor().item<float>();
    float val = output;
    if (toMove == 2) val = -val;
    return val;
}

GNNModel::~GNNModel() {
    delete impl;
}
