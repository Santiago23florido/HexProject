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

/**
 * @brief Configuration for self-play and value training.
 *
 * Sizes are counts; timeLimitMs is in milliseconds and valueScale scales value targets.
 */
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

/**
 * Orchestrates self-play, replay sampling, and value network training.
 *
 * Owns models, optimizer, and buffers; uses internal RNG and optional GPU device.
 */
class RLTrainer {
public:
    /// Constructs the trainer from a configuration.
    explicit RLTrainer(const RLConfig& config);
    /// Runs self-play training and checkpointing; returns 0 on success.
    int run();

private:
    /**
     *Per-move feature snapshot collected during an episode.
     */
    struct EpisodeState {
        std::array<float, 7> features{};
        int toMove{0};
    };

    /// Builds episode features for the given state and player.
    EpisodeState buildEpisodeState(const GameState& state, int playerId) const;
    /// Runs self-play for the requested number of games.
    void collectSelfPlay(int games);
    /// Plays a single game and appends states into episode.
    int playOneGame(std::vector<EpisodeState>& episode, IMoveStrategy& p1, IMoveStrategy& p2, std::mt19937& rng);
    /// Converts an episode into replay samples and adds them to the buffer.
    void addEpisodeToBuffer(const std::vector<EpisodeState>& episode, int winner);

    /// Initializes normalization buffers once enough data is available.
    void maybeInitNormalization();
    /// Performs a number of training updates from the replay buffer.
    void trainUpdates(int updates);
    /// Syncs the evaluation model with the training model.
    void syncEvalModel();
    /// Saves a snapshot of the eval model into the frozen pool.
    void snapshotFrozenModel();
    /// Randomly selects a frozen model, or returns nullptr if none.
    ValueMLP* pickFrozenModel();
    /// Saves the current model checkpoint to disk.
    void saveCheckpoint() const;
    /// Exports the current model to TorchScript.
    void exportTorchScript() const;
    /// Runs a quick TorchScript load-and-forward test.
    void smokeTestTorchScript() const;

    /// Evaluates features using the current eval model.
    float evalFeatures(const std::array<float, 7>& features);
    /// Evaluates features using the provided model.
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
