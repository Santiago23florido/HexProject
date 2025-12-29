#include "gnn/GNNModel.hpp"

#include <filesystem>
#include <iostream>
#include <vector>
#include <cassert>

#include <torch/script.h>
#include <torch/torch.h>

struct GNNModel::Impl {
    torch::jit::script::Module module;
    torch::Device device{torch::kCPU};
    bool useCuda{false};
};

GNNModel::GNNModel(const std::string& modelPath, bool preferCuda) {
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

        //verification of cuda availability
        bool cudaAvailable = torch::cuda::is_available();
    
        impl->useCuda = preferCuda && cudaAvailable;
        impl->device = impl->useCuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
        impl->module = torch::jit::load(path.string());
        impl->module.to(impl->device);
        impl->module.eval();

        std::cout << "[GNN] Device set to: " << (impl->useCuda ? "GPU (CUDA)" : "CPU") << "\n";
        if (preferCuda && !cudaAvailable){
            std::cout << "[Warning] GPU requested but not available. Falling back to CPU.\n";
        }
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
    const int64_t expectedSize = static_cast<int64_t>(batch.numNodes) * static_cast<int64_t>(batch.featureDim);
    if (expectedSize > 0 && static_cast<int64_t>(batch.nodeFeatures.size()) != expectedSize) {
        std::cerr << "[GNN] Feature size mismatch | featureDim=" << batch.featureDim
                  << " numNodes=" << batch.numNodes
                  << " got=" << batch.nodeFeatures.size()
                  << " expected=" << expectedSize << "\n";
    }
    assert(batch.featureDim > 0);
    if (batch.numNodes > 0) {
        assert(static_cast<int64_t>(batch.nodeFeatures.size()) >= expectedSize);
    }

    torch::Tensor x = torch::from_blob(const_cast<float*>(batch.nodeFeatures.data()),
                                       {batch.numNodes, batch.featureDim},
                                       torch::TensorOptions().dtype(torch::kFloat32)).to(impl->device);
    if (batch.featureDim >= 10) {
        const float p1 = (toMove == 1 ? 1.0f : 0.0f);
        const float p2 = (toMove == 2 ? 1.0f : 0.0f);
        x.select(1, 8).fill_(p1);
        x.select(1, 9).fill_(p2);
    }
    const auto tensorDim = x.size(1);
    assert(tensorDim == batch.featureDim);

    // Edge index: two rows, E columns, owning its memory
    torch::Tensor src = torch::tensor(batch.edgeSrc, torch::TensorOptions().dtype(torch::kInt64)).to(impl->device);
    torch::Tensor dst = torch::tensor(batch.edgeDst, torch::TensorOptions().dtype(torch::kInt64)).to(impl->device);
    torch::Tensor edge_index = torch::stack({src, dst}, 0);

    auto output = impl->module.forward({x, edge_index}).toTensor();
    if (!evalLogged) {
        evalLogged = true;
        std::cout << "[GNN] Eval info | device=" << (impl->useCuda ? "CUDA" : "CPU")
                  << " featureDim=" << batch.featureDim
                  << " tensorDim=" << tensorDim
                  << " nodes=" << batch.numNodes
                  << " edges=" << batch.edgeSrc.size()
                  << " output=" << output.item<float>() << "\n";
    }
    float val = output.item<float>();
    return val;
}

GNNModel::~GNNModel() {
    delete impl;
}
