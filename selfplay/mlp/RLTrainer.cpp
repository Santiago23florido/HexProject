#include "RLTrainer.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

#include <torch/nn/utils/clip_grad.h>
#include <torch/script.h>

// Implements the RL trainer that runs self-play, trains the value model, and manages checkpoints/exports.

namespace {

unsigned makeSeed() {
    // Mix steady-clock ticks and random_device output for a seed.
    return static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count() ^
        static_cast<unsigned>(std::random_device{}()));
}

void ensureDir(const std::string& path) {
    // Ensures the parent directory exists for the given path.
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
}

void copyModuleState(const ValueMLP& src, ValueMLP& dst, const torch::Device& device) {
    // Copies parameters and buffers from src to dst on the target device.
    torch::NoGradGuard no_grad;
    auto srcParams = src->named_parameters();
    auto dstParams = dst->named_parameters();
    for (const auto& item : srcParams) {
        auto* target = dstParams.find(item.key());
        if (target) {
            target->copy_(item.value().to(device));
        }
    }

    auto srcBuffers = src->named_buffers();
    auto dstBuffers = dst->named_buffers();
    for (const auto& item : srcBuffers) {
        auto* target = dstBuffers.find(item.key());
        if (target) {
            target->copy_(item.value().to(device));
        }
    }
}

} // namespace

RLTrainer::RLTrainer(const RLConfig& config)
    : cfg_(config),
      device_(torch::kCPU),
      model_(ValueMLP(7, 128, 2, cfg_.valueScale)),
      evalModel_(ValueMLP(7, 128, 2, cfg_.valueScale)),
      lossFn_(torch::nn::SmoothL1LossOptions()),
      buffer_(cfg_.bufferSize),
      rng_() {
    const unsigned seed = makeSeed();
    rng_.seed(seed);
    std::srand(seed);

    if (cfg_.minDepth < 1) cfg_.minDepth = 1;
    if (cfg_.maxDepth < cfg_.minDepth) cfg_.maxDepth = cfg_.minDepth;
    if (cfg_.batchSize == 0) cfg_.batchSize = 1;
    if (cfg_.selfplayThreads < 1) cfg_.selfplayThreads = 1;

    if (cfg_.device == "cuda") {
#if defined(SELFPLAY_HAS_CUDA)
        if (!torch::cuda::is_available()) {
            std::cout << "[RL] CUDA requested but not available. Using CPU.\n";
        } else {
            device_ = torch::kCUDA;
        }
#else
        std::cout << "[RL] CUDA not enabled in this build. Using CPU.\n";
#endif
    }
    model_->to(device_);
    evalModel_->to(evalDevice_);
    evalModel_->eval();
    optimizer_ = std::make_unique<torch::optim::AdamW>(
        model_->parameters(), torch::optim::AdamWOptions(cfg_.lr));
    syncEvalModel();
}

RLTrainer::EpisodeState RLTrainer::buildEpisodeState(const GameState& state, int playerId) const {
    // Feature order matches ValueFeatures fields.
    const ValueFeatures feat = computeValueFeatures(state, playerId);
    EpisodeState ep;
    ep.features = {
        static_cast<float>(feat.distSelf),
        static_cast<float>(feat.distOpp),
        static_cast<float>(feat.libsSelf),
        static_cast<float>(feat.libsOpp),
        static_cast<float>(feat.bridgesSelf),
        static_cast<float>(feat.bridgesOpp),
        static_cast<float>(feat.center),
    };
    ep.toMove = playerId;
    return ep;
}

int RLTrainer::playOneGame(std::vector<EpisodeState>& episode, IMoveStrategy& p1, IMoveStrategy& p2, std::mt19937& rng) {
    Board board(cfg_.boardSize);
    int currentPlayer = 1;
    GameState state(board, currentPlayer);

    while (true) {
        episode.push_back(buildEpisodeState(state, currentPlayer));

        int winner = state.Winner();
        if (winner != 0) return winner;

        auto moves = state.GetAvailableMoves();
        if (moves.empty()) return 0;

        IMoveStrategy& strat = (currentPlayer == 1) ? p1 : p2;
        int move = -1;
        if (cfg_.randomFirstMove && currentPlayer == 1 && episode.size() == 1) {
            std::uniform_int_distribution<std::size_t> dist(0, moves.size() - 1);
            move = moves[dist(rng)];
        } else {
            move = strat.select(state, currentPlayer);
        }
        // Treat invalid moves as a draw to terminate the game.
        if (move < 0 || move >= cfg_.boardSize * cfg_.boardSize) return 0;

        board.place(move, currentPlayer);
        currentPlayer = (currentPlayer == 1 ? 2 : 1);
        state.Update(board, currentPlayer);
    }
}

void RLTrainer::addEpisodeToBuffer(const std::vector<EpisodeState>& episode, int winner) {
    std::vector<ReplaySample> samples;
    samples.reserve(episode.size());
    // Scale targets by valueScale to match the network output.
    for (const auto& entry : episode) {
        int z = (winner == entry.toMove) ? 1 : -1;
        ReplaySample sample;
        sample.features = entry.features;
        sample.target = static_cast<float>(z) * cfg_.valueScale;
        samples.push_back(sample);
    }
    buffer_.addBatch(samples);
}

void RLTrainer::maybeInitNormalization() {
    // Initialize normalization stats from a capped sample of recent data.
    if (normReady_ || buffer_.size() < cfg_.batchSize) return;

    const std::size_t count = std::min<std::size_t>(buffer_.size(), 4096);
    auto sample = buffer_.sample(count, rng_);
    if (sample.empty()) return;

    std::array<double, 7> sum{};
    std::array<double, 7> sumSq{};
    for (const auto& s : sample) {
        for (std::size_t i = 0; i < s.features.size(); ++i) {
            const double v = static_cast<double>(s.features[i]);
            sum[i] += v;
            sumSq[i] += v * v;
        }
    }
    std::vector<float> meanVals(7, 0.0f);
    std::vector<float> stdVals(7, 1.0f);
    for (std::size_t i = 0; i < meanVals.size(); ++i) {
        const double mean = sum[i] / static_cast<double>(count);
        const double var = std::max(1e-6, (sumSq[i] / static_cast<double>(count)) - (mean * mean));
        meanVals[i] = static_cast<float>(mean);
        stdVals[i] = static_cast<float>(std::sqrt(var));
    }

    auto mean = torch::from_blob(meanVals.data(), {7}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto std = torch::from_blob(stdVals.data(), {7}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    // Clone ensures buffers own their data before transfer to device.
    model_->setNormalization(mean.to(device_), std.to(device_));
    normReady_ = true;
    syncEvalModel();
}

void RLTrainer::trainUpdates(int updates) {
    if (!buffer_.canSample(cfg_.batchSize)) return;
    maybeInitNormalization();

    model_->train();
    const std::size_t updatesBefore = totalUpdates_;
    if (!trainingLogged_ && updates > 0) {
        trainingLogged_ = true;
        std::cout << "[RL] Training updates start (single-thread main loop)\n";
    }
    for (int u = 0; u < updates; ++u) {
        if (!buffer_.canSample(cfg_.batchSize)) break;
        auto batch = buffer_.sample(cfg_.batchSize, rng_);
        if (batch.empty()) break;

        auto x = torch::empty({static_cast<long>(batch.size()), 7},
                              torch::TensorOptions().dtype(torch::kFloat32));
        auto y = torch::empty({static_cast<long>(batch.size())},
                              torch::TensorOptions().dtype(torch::kFloat32));
        auto xAcc = x.accessor<float, 2>();
        auto yAcc = y.accessor<float, 1>();
        for (std::size_t i = 0; i < batch.size(); ++i) {
            for (int j = 0; j < 7; ++j) {
                xAcc[static_cast<long>(i)][j] = batch[i].features[static_cast<std::size_t>(j)];
            }
            yAcc[static_cast<long>(i)] = batch[i].target;
        }

        auto xDev = x.to(device_);
        auto yDev = y.to(device_);
        auto pred = model_->forward(xDev);
        auto loss = lossFn_(pred, yDev);

        optimizer_->zero_grad();
        loss.backward();
        // Clip gradients to stabilize training.
        torch::nn::utils::clip_grad_norm_(model_->parameters(), cfg_.gradClip);
        optimizer_->step();

        runningLoss_ += loss.item<double>();
        totalUpdates_++;
    }
    model_->eval();
    if (totalUpdates_ > updatesBefore) {
        syncEvalModel();
    }
}

float RLTrainer::evalFeatures(const std::array<float, 7>& features) {
    return evalFeaturesWithModel(features, evalModel_);
}

float RLTrainer::evalFeaturesWithModel(const std::array<float, 7>& features, ValueMLP& model) {
    // No gradients required for inference.
    torch::NoGradGuard no_grad;
    auto x = torch::from_blob(const_cast<float*>(features.data()),
                              {1, 7},
                              torch::TensorOptions().dtype(torch::kFloat32));
    auto out = model->forward(x);
    return out.item<float>();
}

void RLTrainer::syncEvalModel() {
    // Keep eval model in sync with the training model.
    std::lock_guard<std::mutex> lock(evalModelMutex_);
    copyModuleState(model_, evalModel_, evalDevice_);
    evalModel_->eval();
}

void RLTrainer::snapshotFrozenModel() {
    if (cfg_.maxFrozen == 0) return;

    std::lock_guard<std::mutex> evalLock(evalModelMutex_);
    std::lock_guard<std::mutex> frozenLock(frozenMutex_);
    ValueMLP frozen(7, 128, model_->depth(), cfg_.valueScale);
    frozen->to(evalDevice_);
    copyModuleState(evalModel_, frozen, evalDevice_);
    frozen->eval();
    frozenPool_.push_back(frozen);
    if (frozenPool_.size() > cfg_.maxFrozen) {
        frozenPool_.erase(frozenPool_.begin());
    }
    std::cout << "[RL] Frozen snapshot stored | pool=" << frozenPool_.size() << "\n";
}

ValueMLP* RLTrainer::pickFrozenModel() {
    std::lock_guard<std::mutex> lock(frozenMutex_);
    if (frozenPool_.empty()) return nullptr;
    std::uniform_int_distribution<std::size_t> dist(0, frozenPool_.size() - 1);
    return &frozenPool_[dist(rng_)];
}

void RLTrainer::collectSelfPlay(int games) {
    std::uniform_int_distribution<int> depthDist(cfg_.minDepth, cfg_.maxDepth);

    float probFrozen = std::clamp(cfg_.probFrozen, 0.0f, 1.0f);
    float probHeuristic = std::clamp(cfg_.probHeuristic, 0.0f, 1.0f);
    const float probSum = probFrozen + probHeuristic;
    if (probSum > 1.0f) {
        probFrozen /= probSum;
        probHeuristic /= probSum;
    }

    if (cfg_.snapshotEvery > 0 && cfg_.maxFrozen > 0) {
        snapshotFrozenModel();
    }

    const int threadCount = std::max(1, cfg_.selfplayThreads);
    std::cout << "[RL] Self-play threads=" << threadCount
              << " | training updates run on main thread\n";
    // Single-thread mode runs self-play and training updates in the same loop.
    if (threadCount <= 1) {
        auto evalFn = [this](const std::array<float, 7>& f) { return evalFeatures(f); };

        for (int i = 0; i < games; ++i) {
            int depthP1 = depthDist(rng_);
            int depthP2 = depthDist(rng_);

            NegamaxStrategy p1(depthP1, cfg_.timeLimitMs, "", false, false, cfg_.alpha, false);
            p1.setMlpEvaluator(evalFn);
            p1.setEvalMixAlpha(cfg_.alpha);
            p1.setLogUsage(false);

            std::string opponentLabel = "current";
            bool useHeuristic = false;
            bool useFrozen = false;
            if (probFrozen > 0.0f || probHeuristic > 0.0f) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float r = dist(rng_);
                if (r < probHeuristic) {
                    useHeuristic = true;
                    opponentLabel = "heuristic";
                } else if (r < (probHeuristic + probFrozen)) {
                    useFrozen = true;
                    opponentLabel = "frozen";
                }
            }

            std::unique_ptr<NegamaxStrategy> p2Model;
            std::unique_ptr<NegamaxHeuristicStrategy> p2Heur;
            IMoveStrategy* p2 = nullptr;
            if (useHeuristic) {
                p2Heur = std::make_unique<NegamaxHeuristicStrategy>(depthP2, cfg_.timeLimitMs);
                p2Heur->setLogUsage(false);
                p2 = p2Heur.get();
            } else {
                p2Model = std::make_unique<NegamaxStrategy>(depthP2, cfg_.timeLimitMs, "", false, false, cfg_.alpha, false);
                if (useFrozen) {
                    ValueMLP* frozen = pickFrozenModel();
                    if (frozen) {
                        auto frozenEvalFn = [this, frozen](const std::array<float, 7>& f) {
                            return evalFeaturesWithModel(f, *frozen);
                        };
                        p2Model->setMlpEvaluator(frozenEvalFn);
                    } else {
                        opponentLabel = "current";
                        p2Model->setMlpEvaluator(evalFn);
                    }
                } else {
                    p2Model->setMlpEvaluator(evalFn);
                }
                p2Model->setEvalMixAlpha(cfg_.alpha);
                p2Model->setLogUsage(false);
                p2 = p2Model.get();
            }

            std::vector<EpisodeState> episode;
            int winner = playOneGame(episode, p1, *p2, rng_);
            addEpisodeToBuffer(episode, winner);

            if (winner == 1) winP1_++;
            else if (winner == 2) winP2_++;
            else draws_++;

            trainUpdates(cfg_.updatesPerGame);

            std::cout << "[RL] Game " << (i + 1) << "/" << games
                      << " winner=" << winner
                      << " opponent=" << opponentLabel
                      << " buffer=" << buffer_.size()
                      << "\n"
                      << std::flush;

            if (cfg_.checkpointEvery > 0 && (i + 1) % cfg_.checkpointEvery == 0) {
                saveCheckpoint();
                if (cfg_.exportTs) {
                    exportTorchScript();
                    smokeTestTorchScript();
                }
            }

            if (cfg_.reportEvery > 0 && (i + 1) % cfg_.reportEvery == 0) {
                const double avgLoss = totalUpdates_ > 0 ? (runningLoss_ / static_cast<double>(totalUpdates_)) : 0.0;
                const int totalGames = winP1_ + winP2_ + draws_;
                const double winrate = totalGames > 0 ? (static_cast<double>(winP1_) / totalGames) : 0.0;
                std::cout << "[RL] Game " << (i + 1)
                          << " | buffer=" << buffer_.size()
                          << " | updates=" << totalUpdates_
                          << " | avgLoss=" << std::fixed << std::setprecision(6) << avgLoss
                          << " | winrateP1=" << std::setprecision(3) << winrate
                          << "\n"
                          << std::flush;
            }

            if (cfg_.snapshotEvery > 0 && (i + 1) % cfg_.snapshotEvery == 0) {
                snapshotFrozenModel();
            }
        }
        return;
    }

    struct GameResult {
        std::vector<EpisodeState> episode;
        int winner{0};
        std::string opponentLabel;
    };

    std::atomic<int> nextGame{0};
    std::atomic<int> workersDone{0};
    std::mutex queueMutex;
    std::condition_variable queueCv;
    std::deque<GameResult> queue;

    auto workerFn = [&](int workerId) {
        std::mt19937 localRng(makeSeed() ^ static_cast<unsigned>(workerId + 1));
        std::uniform_int_distribution<int> localDepth(cfg_.minDepth, cfg_.maxDepth);
        std::uniform_real_distribution<float> localProb(0.0f, 1.0f);

        // Each worker keeps its own eval copies to avoid locking during forward passes.
        ValueMLP currentEval(7, 128, model_->depth(), cfg_.valueScale);
        currentEval->to(evalDevice_);
        ValueMLP frozenEval(7, 128, model_->depth(), cfg_.valueScale);
        frozenEval->to(evalDevice_);

        for (;;) {
            int idx = nextGame.fetch_add(1);
            if (idx >= games) break;

            {
                std::lock_guard<std::mutex> lock(evalModelMutex_);
                copyModuleState(evalModel_, currentEval, evalDevice_);
            }
            auto evalFn = [this, &currentEval](const std::array<float, 7>& f) {
                return evalFeaturesWithModel(f, currentEval);
            };

            bool useHeuristic = false;
            bool useFrozen = false;
            std::string opponentLabel = "current";
            if (probFrozen > 0.0f || probHeuristic > 0.0f) {
                float r = localProb(localRng);
                if (r < probHeuristic) {
                    useHeuristic = true;
                    opponentLabel = "heuristic";
                } else if (r < (probHeuristic + probFrozen)) {
                    useFrozen = true;
                    opponentLabel = "frozen";
                }
            }

            int depthP1 = localDepth(localRng);
            int depthP2 = localDepth(localRng);

            NegamaxStrategy p1(depthP1, cfg_.timeLimitMs, "", false, false, cfg_.alpha, false);
            p1.setMlpEvaluator(evalFn);
            p1.setEvalMixAlpha(cfg_.alpha);
            p1.setLogUsage(false);

            std::unique_ptr<NegamaxStrategy> p2Model;
            std::unique_ptr<NegamaxHeuristicStrategy> p2Heur;
            IMoveStrategy* p2 = nullptr;
            if (useHeuristic) {
                p2Heur = std::make_unique<NegamaxHeuristicStrategy>(depthP2, cfg_.timeLimitMs);
                p2Heur->setLogUsage(false);
                p2 = p2Heur.get();
            } else {
                p2Model = std::make_unique<NegamaxStrategy>(depthP2, cfg_.timeLimitMs, "", false, false, cfg_.alpha, false);
                if (useFrozen) {
                    bool haveFrozen = false;
                    {
                        std::lock_guard<std::mutex> lock(frozenMutex_);
                        if (!frozenPool_.empty()) {
                            std::uniform_int_distribution<std::size_t> frozenDist(0, frozenPool_.size() - 1);
                            copyModuleState(frozenPool_[frozenDist(localRng)], frozenEval, evalDevice_);
                            haveFrozen = true;
                        }
                    }
                    if (haveFrozen) {
                        auto frozenEvalFn = [this, &frozenEval](const std::array<float, 7>& f) {
                            return evalFeaturesWithModel(f, frozenEval);
                        };
                        p2Model->setMlpEvaluator(frozenEvalFn);
                    } else {
                        opponentLabel = "current";
                        p2Model->setMlpEvaluator(evalFn);
                    }
                } else {
                    p2Model->setMlpEvaluator(evalFn);
                }
                p2Model->setEvalMixAlpha(cfg_.alpha);
                p2Model->setLogUsage(false);
                p2 = p2Model.get();
            }

            std::vector<EpisodeState> episode;
            int winner = playOneGame(episode, p1, *p2, localRng);

            GameResult result;
            result.episode = std::move(episode);
            result.winner = winner;
            result.opponentLabel = opponentLabel;
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                queue.push_back(std::move(result));
            }
            queueCv.notify_one();
        }

        workersDone.fetch_add(1);
        queueCv.notify_one();
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threadCount));
    for (int i = 0; i < threadCount; ++i) {
        workers.emplace_back(workerFn, i);
    }

    int processed = 0;
    while (processed < games) {
        GameResult result;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCv.wait(lock, [&]() {
                return !queue.empty() || workersDone.load() == threadCount;
            });
            if (queue.empty()) {
                if (workersDone.load() == threadCount) break;
                continue;
            }
            result = std::move(queue.front());
            queue.pop_front();
        }

        processed++;
        addEpisodeToBuffer(result.episode, result.winner);

        if (result.winner == 1) winP1_++;
        else if (result.winner == 2) winP2_++;
        else draws_++;

        trainUpdates(cfg_.updatesPerGame);

        std::cout << "[RL] Game " << processed << "/" << games
                  << " winner=" << result.winner
                  << " opponent=" << result.opponentLabel
                  << " buffer=" << buffer_.size()
                  << "\n"
                  << std::flush;

        if (cfg_.checkpointEvery > 0 && processed % cfg_.checkpointEvery == 0) {
            saveCheckpoint();
            if (cfg_.exportTs) {
                exportTorchScript();
                smokeTestTorchScript();
            }
        }

        if (cfg_.reportEvery > 0 && processed % cfg_.reportEvery == 0) {
            const double avgLoss = totalUpdates_ > 0 ? (runningLoss_ / static_cast<double>(totalUpdates_)) : 0.0;
            const int totalGames = winP1_ + winP2_ + draws_;
            const double winrate = totalGames > 0 ? (static_cast<double>(winP1_) / totalGames) : 0.0;
            std::cout << "[RL] Game " << processed
                      << " | buffer=" << buffer_.size()
                      << " | updates=" << totalUpdates_
                      << " | avgLoss=" << std::fixed << std::setprecision(6) << avgLoss
                      << " | winrateP1=" << std::setprecision(3) << winrate
                      << "\n"
                      << std::flush;
        }

        if (cfg_.snapshotEvery > 0 && processed % cfg_.snapshotEvery == 0) {
            snapshotFrozenModel();
        }
    }

    for (auto& worker : workers) {
        if (worker.joinable()) worker.join();
    }
}

void RLTrainer::saveCheckpoint() const {
    // Writes the model checkpoint to disk.
    ensureDir(cfg_.checkpointPath);
    torch::serialize::OutputArchive archive;
    model_->save(archive);
    archive.save_to(cfg_.checkpointPath);
    std::cout << "[RL] Saved checkpoint to " << cfg_.checkpointPath << "\n";
}

void RLTrainer::exportTorchScript() const {
    // Writes a TorchScript export of the current model.
    ensureDir(cfg_.exportPath);
    std::string error;
    if (!saveValueMLPTorchScript(model_, cfg_.exportPath, &error)) {
        std::cerr << "[RL] TorchScript export failed: " << error << "\n";
        return;
    }
    std::cout << "[RL] Exported TorchScript to " << cfg_.exportPath << "\n";
}

void RLTrainer::smokeTestTorchScript() const {
    // Best-effort load-and-run test for the exported TorchScript.
    try {
        torch::jit::Module module = torch::jit::load(cfg_.exportPath);
        module.eval();
        auto x = torch::zeros({1, 7}, torch::TensorOptions().dtype(torch::kFloat32));
        auto out = module.forward({x}).toTensor();
        std::cout << "[RL] TorchScript smoke test output: " << out.item<float>() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[RL] TorchScript smoke test failed: " << e.what() << "\n";
    }
}

int RLTrainer::run() {
    // Runs the configured training loop and handles optional exports.
    std::cout << "[RL] Training on device: " << (device_.is_cuda() ? "CUDA" : "CPU") << "\n";
    collectSelfPlay(cfg_.trainGames);
    saveCheckpoint();
    if (cfg_.exportTs) {
        exportTorchScript();
        smokeTestTorchScript();
    }
    return 0;
}
