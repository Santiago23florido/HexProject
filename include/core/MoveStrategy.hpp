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

class IMoveStrategy{
    public:
        virtual int select (const GameState& state,int playerId) = 0;
        virtual ~IMoveStrategy() = default;
};

class RandomStrategy : public IMoveStrategy {
public:
    int select(const GameState& state, int playerId) override;
};


class MonteCarloStrategy : public IMoveStrategy {
    int SimulationsPerMove;
public:
    MonteCarloStrategy(int sims);
    int select(const GameState& state, int playerId) override; 
    int simulate(GameState state, int playerId); 
};

class Zobrist {
public:
    Zobrist() = default;
    Zobrist(const Zobrist&) = default;
    ~Zobrist() = default;

    explicit Zobrist(int boardSize);

    uint64_t computeHash(const Board& board) const;
    uint64_t applyMoveHash(uint64_t hash, int moveIndex, int color) const;
    uint64_t undoMoveHash(uint64_t hash, int moveIndex, int color) const;

private:
    int boardSize{0};
    std::vector<std::array<uint64_t, 2>> keys; 
    uint64_t side{0};
};

struct SearchResult{
    int bestMove;
    int score;
    bool completed{true};
    bool failLow{false};
    bool failHigh{false};
};

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

ValueFeatures computeValueFeatures(const GameState& state, int playerId);


enum class TTFlag { EXACT, LOWER, UPPER };

struct TTEntry {
    uint64_t key{0};
    int depth{0};
    int value{0};
    TTFlag flag{TTFlag::EXACT};
    int bestMove{-1};
};


class NegamaxStrategy : public IMoveStrategy {
public:
    NegamaxStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath = "scripts/models/hex_value_ts_mp.pt", bool heuristicOnly = false, bool preferCuda = false, float evalMixAlpha = 0.2f, bool loadModel = true);
    int select(const GameState& state, int playerId) override;
    int getmaxDepth(const NegamaxStrategy& strat)const;
    void setMlpEvaluator(std::function<float(const std::array<float, 7>&)> evaluator);
    void setEvalMixAlpha(float alpha);
    void setLogUsage(bool enable);
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


class NegamaxHeuristicStrategy : public NegamaxStrategy {
public:
    NegamaxHeuristicStrategy(int maxDepth, int timeLimitMs);
};


class NegamaxGnnStrategy : public NegamaxStrategy {
public:
    NegamaxGnnStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath = "scripts/models/hex_value_ts_mp.pt", bool preferCuda = false, float evalMixAlpha = 0.2f);
};
