#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>
#include <iostream>
#include <chrono>
#include "core/GameState.hpp"
#include "core/Board.hpp"
#include "gnn/FeatureExtractor.hpp"
#include "gnn/GNNModel.hpp"

/**
 *Strategy interface for selecting a move.
 */
class IMoveStrategy{
    public:
        /// Returns a selected move as a linear index.
        virtual int select (const GameState& state,int playerId) = 0;
        /// Virtual destructor for safe polymorphic cleanup.
        virtual ~IMoveStrategy() = default;
};

/**
 *  Random move selection strategy.
 */
class RandomStrategy : public IMoveStrategy {
public:
    /// Selects a random legal move.
    int select(const GameState& state, int playerId) override;
};


/**
 *  Monte Carlo rollout strategy.
 */
class MonteCarloStrategy : public IMoveStrategy {
    int SimulationsPerMove;
public:
    /// Creates a strategy with sims rollouts per move.
    MonteCarloStrategy(int sims);
    /// Selects the move with the best rollout win count.
    int select(const GameState& state, int playerId) override; 
    /// Simulates a random playout and returns the winner id.
    int simulate(GameState state, int playerId); 
};

/**
 *  Zobrist hashing helper for board states.
 */
class Zobrist {
public:
    Zobrist() = default;
    Zobrist(const Zobrist&) = default;
    ~Zobrist() = default;

    /// Creates hash keys for a board with boardSize cells.
    explicit Zobrist(int boardSize);

    /// Computes a hash for the current board.
    uint64_t computeHash(const Board& board) const;
    /// Updates a hash by applying a move at moveIndex for color.
    uint64_t applyMoveHash(uint64_t hash, int moveIndex, int color) const;
    /// Updates a hash by undoing a move at moveIndex for color.
    uint64_t undoMoveHash(uint64_t hash, int moveIndex, int color) const;

private:
    int boardSize{0};
    std::vector<std::array<uint64_t, 2>> keys; 
    uint64_t side{0};
};

/**
 *  Negamax search result container.
 */
struct SearchResult{
    int bestMove;
    int score;
    bool completed{true};
    bool failLow{false};
    bool failHigh{false};
};

/**
 * Feature summary used by heuristic and MLP evaluators.
 */
struct ValueFeatures {
    int N{0};
    int distSelf{0};
    int distOpp{0};
    int libsSelf{0};
    int libsOpp{0};
    int bridgesSelf{0};
    int bridgesOpp{0};
    int center{0};
};

/// Computes value features for the given player.
ValueFeatures computeValueFeatures(const GameState& state, int playerId);


/// Transposition table entry flag.
enum class TTFlag { EXACT, LOWER, UPPER };

/**
 * Transposition table entry.
 */
struct TTEntry {
    uint64_t key{0};
    int depth{0};
    int value{0};
    TTFlag flag{TTFlag::EXACT};
    int bestMove{-1};
};


/**
 * Negamax search strategy with heuristic/MLP/GNN evaluation.
 *
 * timeLimitMs is in milliseconds.
 */
class NegamaxStrategy : public IMoveStrategy {
public:
    /// Creates a Negamax strategy with optional model evaluation.
    NegamaxStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath = "scripts/models/hex_value_ts_mp.pt", bool heuristicOnly = false, bool preferCuda = false, float evalMixAlpha = 0.2f, bool loadModel = true);
    /// Selects the best move for the given state and player.
    int select(const GameState& state, int playerId) override;
    /// Returns the maximum search depth.
    int getmaxDepth(const NegamaxStrategy& strat)const;
    /// Sets an external MLP evaluator for feature inputs.
    void setMlpEvaluator(std::function<float(const std::array<float, 7>&)> evaluator);
    /// Sets the blend factor between heuristic and model scores.
    void setEvalMixAlpha(float alpha);
    /// Enables or disables usage logging.
    void setLogUsage(bool enable);
    /// Sets the number of parallel root threads.
    void setParallelThreads(int threads);

private:
    float mlEvalFromFeatures(const GameState& state, int playerId) const;
    SearchResult negamaxParallelRoot(const GameState& state, int depth, int alpha, int beta, int playerId, std::chrono::steady_clock::time_point startTime) const;
    SearchResult iterativeDeepening(const GameState& state, int playerId) const;
    SearchResult negamax(const GameState& state, int depth, int alpha, int beta, int playerId, std::chrono::steady_clock::time_point startTime) const;
    int maxDepth;
    int timeLimitMs;
    int valueScale{1000}; 
    bool useHeuristic{false};
    float evalMixAlpha{0.2f};
    bool logUsage{true};
    int parallelRootThreads{1};
    mutable bool usageLogged{false};
    mutable int rootPlayer{1};
    mutable int lastEvalPlayer{0};
    mutable float lastEvalVal{0.0f};
    mutable int lastEvalScaled{0};
    static constexpr int MAX_DEPTH = 64;
    static constexpr std::size_t TT_SIZE = 1u << 18;
    mutable std::vector<TTEntry> transposition; 
    mutable std::vector<std::array<int, 2>> killers; 
    mutable std::vector<int> history;                
    mutable Zobrist zobrist;
    mutable int zobristCells{0};
    FeatureExtractor extractor;
    mutable FeatureBatch gnnBatch;
    GNNModel model;
    std::function<float(const std::array<float, 7>&)> mlpEvaluator;
};


/**
 * Negamax strategy that forces heuristic evaluation.
 */
class NegamaxHeuristicStrategy : public NegamaxStrategy {
public:
    /// Creates a heuristic-only Negamax strategy.
    NegamaxHeuristicStrategy(int maxDepth, int timeLimitMs);
};


/**
 * Negamax strategy that evaluates with a GNN/MLP TorchScript model.
 */
class NegamaxGnnStrategy : public NegamaxStrategy {
public:
    /// Creates a Negamax strategy configured for TorchScript evaluation.
    NegamaxGnnStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath = "scripts/models/hex_value_ts_mp.pt", bool preferCuda = false, float evalMixAlpha = 0.2f);
};
