#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "ReplayBuffer.hpp"
#include "ValueMLP.hpp"
#include "core/MoveStrategy.hpp"

struct RLConfig {
    int boardSize{7};
    int minDepth{10};
    int maxDepth{20};
    int timeLimitMs{10000};
    int trainGames{200};
    std::size_t bufferSize{50000};
    std::size_t batchSize{256};
    int updatesPerGame{1};
    int reportEvery{10};
    int checkpointEvery{10};
    int snapshotEvery{20};
    int selfplayThreads{1};
    float alpha{0.2f};
    float lr{3e-4f};
    float gradClip{1.0f};
    float valueScale{10000.0f};
    float probFrozen{0.3f};
    float probHeuristic{0.0f};
    std::size_t maxFrozen{5};
    std::string device{"cuda"};
    std::string checkpointPath{"scripts/models/value_mlp_state.pt"};
    std::string exportPath{"scripts/models/hex_value_ts_mp.pt"};
    bool exportTs{false};
    bool randomFirstMove{true};
};

class RLTrainer {
public:
    explicit RLTrainer(const RLConfig& config);
    int run();

private:
    struct EpisodeState {
        std::array<float, 7> features{};
        int toMove{0};
    };

    EpisodeState buildEpisodeState(const GameState& state, int playerId) const;
    void collectSelfPlay(int games);
    int playOneGame(std::vector<EpisodeState>& episode, IMoveStrategy& p1, IMoveStrategy& p2, std::mt19937& rng);
    void addEpisodeToBuffer(const std::vector<EpisodeState>& episode, int winner);

    void maybeInitNormalization();
    void trainUpdates(int updates);
    void syncEvalModel();
    void snapshotFrozenModel();
    ValueMLP* pickFrozenModel();
    void saveCheckpoint() const;
    void exportTorchScript() const;
    void smokeTestTorchScript() const;

    float evalFeatures(const std::array<float, 7>& features);
    float evalFeaturesWithModel(const std::array<float, 7>& features, ValueMLP& model);

    RLConfig cfg_;
    torch::Device device_;
    ValueMLP model_;
    ValueMLP evalModel_;
    torch::Device evalDevice_{torch::kCPU};
    std::vector<ValueMLP> frozenPool_;
    std::unique_ptr<torch::optim::AdamW> optimizer_;
    torch::nn::SmoothL1Loss lossFn_;
    ReplayBuffer buffer_;
    std::mt19937 rng_;
    std::mutex evalModelMutex_;
    std::mutex frozenMutex_;
    bool normReady_{false};
    bool trainingLogged_{false};
    std::size_t totalUpdates_{0};
    double runningLoss_{0.0};
    int winP1_{0};
    int winP2_{0};
    int draws_{0};
};
