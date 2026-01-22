#include "gnn/GNNModel.hpp"

#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>

#include <torch/script.h>
#include <torch/torch.h>

struct GNNModel::Impl {
    torch::jit::script::Module module;
    torch::Device device{torch::kCPU};
    bool useCuda{false};
    int forwardInputCount{0};
    std::unordered_map<int, torch::Tensor> edgeIndexCache;
};

namespace {
struct CacheKey {
    std::string path;
    bool useCuda{false};

    bool operator==(const CacheKey& other) const {
        return useCuda == other.useCuda && path == other.path;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const {
        std::size_t h1 = std::hash<std::string>{}(key.path);
        std::size_t h2 = std::hash<bool>{}(key.useCuda);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

std::mutex g_cacheMutex;
std::unordered_map<CacheKey, std::weak_ptr<void>, CacheKeyHash> g_modelCache;
} // namespace

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
        throw std::runtime_error(std::string("GNN model file not found at ") + path.string());
    }

    try {
        //verification of cuda availability
        bool cudaAvailable = torch::cuda::is_available();
    
        const bool useCuda = preferCuda && cudaAvailable;
        const CacheKey cacheKey{path.string(), useCuda};

        {
            std::lock_guard<std::mutex> lock(g_cacheMutex);
            auto it = g_modelCache.find(cacheKey);
            if (it != g_modelCache.end()) {
                auto cached = it->second.lock();
                if (cached) {
                    impl = std::static_pointer_cast<Impl>(cached);
                    loaded = true;
                    std::cout << "[GNN] Device set to: " << (impl->useCuda ? "GPU (CUDA)" : "CPU") << "\n";
                    if (preferCuda && !cudaAvailable){
                        std::cout << "[Warning] GPU requested but not available. Falling back to CPU.\n";
                    }
                    return;
                }
                g_modelCache.erase(it);
            }
        }

        impl = std::make_shared<Impl>();
        impl->useCuda = useCuda;
        impl->device = impl->useCuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
        impl->module = torch::jit::load(path.string());
        impl->module.to(impl->device);
        impl->module.eval();
        {
            auto method = impl->module.get_method("forward");
            const auto& schema = method.function().getSchema();
            const int argCount = static_cast<int>(schema.arguments().size());
            impl->forwardInputCount = (argCount > 0 ? argCount - 1 : 0);
        }

        {
            std::lock_guard<std::mutex> lock(g_cacheMutex);
            g_modelCache.emplace(cacheKey, std::static_pointer_cast<void>(impl));
        }

        std::cout << "[GNN] Device set to: " << (impl->useCuda ? "GPU (CUDA)" : "CPU") << "\n";
        if (preferCuda && !cudaAvailable){
            std::cout << "[Warning] GPU requested but not available. Falling back to CPU.\n";
        }
        loaded = true;
    } catch (const std::exception& e) {
        loaded = false;
        impl.reset();
        throw std::runtime_error(std::string("Failed to load TorchScript model: ") + e.what());
    }
}

bool GNNModel::isLoaded() const {
    return loaded;
}

bool GNNModel::usesCuda() const {
    return loaded && impl != nullptr && impl->useCuda;
}

bool GNNModel::expectsEdgeIndex() const {
    return loaded && impl != nullptr && impl->forwardInputCount >= 2;
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

    // Edge index: two rows, E columns, owning its memory (cache per board size).
    const int cacheKey = (batch.N > 0 ? batch.N : batch.numNodes);
    torch::Tensor edge_index;
    auto it = impl->edgeIndexCache.find(cacheKey);
    if (it != impl->edgeIndexCache.end()) {
        edge_index = it->second;
    } else {
        torch::Tensor src = torch::tensor(batch.edgeSrc, torch::TensorOptions().dtype(torch::kInt64)).to(impl->device);
        torch::Tensor dst = torch::tensor(batch.edgeDst, torch::TensorOptions().dtype(torch::kInt64)).to(impl->device);
        edge_index = torch::stack({src, dst}, 0);
        impl->edgeIndexCache.emplace(cacheKey, edge_index);
    }

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

float GNNModel::evaluateFeatures(const std::array<float, 7>& features) const {
    if (!loaded || impl == nullptr) return 0.0f;
    if (impl->forwardInputCount < 1) return 0.0f;

    torch::NoGradGuard no_grad;
    const int inputDim = static_cast<int>(features.size());
    struct ThreadLocalBuffers {
        torch::Tensor cpu;
        torch::Tensor dev;
        int dim{0};
        bool useCuda{false};
        int deviceIndex{-1};
    };
    thread_local ThreadLocalBuffers tls;
    const int deviceIndex = impl->useCuda ? impl->device.index() : -1;
    if (!tls.cpu.defined() || tls.dim != inputDim || tls.useCuda != impl->useCuda || tls.deviceIndex != deviceIndex) {
        tls.cpu = torch::zeros({1, inputDim}, torch::TensorOptions().dtype(torch::kFloat32));
        tls.dim = inputDim;
        tls.useCuda = impl->useCuda;
        tls.deviceIndex = deviceIndex;
        if (impl->useCuda) {
            tls.dev = torch::zeros({1, inputDim}, torch::TensorOptions().dtype(torch::kFloat32).device(impl->device));
        } else {
            tls.dev = torch::Tensor();
        }
    }

    auto* data = tls.cpu.data_ptr<float>();
    for (int i = 0; i < inputDim; ++i) {
        data[i] = features[static_cast<std::size_t>(i)];
    }

    torch::Tensor x = tls.cpu;
    if (impl->useCuda) {
        tls.dev.copy_(tls.cpu);
        x = tls.dev;
    }
    auto output = impl->module.forward({x}).toTensor();
    float val = output.item<float>();
    return val;
}

GNNModel::~GNNModel() = default;
