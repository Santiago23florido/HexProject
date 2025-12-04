#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <vector>
#include <iostream>
#include <chrono>
#include "GameState.hpp"
#include "Board.hpp"
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
    std::vector<std::array<uint64_t, 2>> keys; // keys[cell][color]
    uint64_t side{0};
};

struct SearchResult{
    int bestMove;
    int score;
    bool completed{true};
    bool failLow{false};
    bool failHigh{false};
};

// Transposition table entry for negamax
enum class TTFlag { EXACT, LOWER, UPPER };

struct TTEntry {
    uint64_t key{0};
    int depth{0};
    int value{0};
    TTFlag flag{TTFlag::EXACT};
    int bestMove{-1};
};

// Simple negamax-based strategy interface
class NegamaxStrategy : public IMoveStrategy {
public:
    NegamaxStrategy(int maxDepth, int timeLimitMs, const std::string& modelPath = "models/hex_value_ts.pt", bool heuristicOnly = false);
    int select(const GameState& state, int playerId) override;
    int getmaxDepth(const NegamaxStrategy& strat)const;

private:
    SearchResult iterativeDeepening(const GameState& state, int playerId) const;
    SearchResult negamax(const GameState& state, int depth, int alpha, int beta, int playerId, std::chrono::steady_clock::time_point startTime) const;
    int maxDepth;
    int timeLimitMs;
    int valueScale{1000}; // scale GNN output to search score space
    bool useHeuristic{false};
    mutable bool usageLogged{false};
    mutable int rootPlayer{1};
    mutable int lastEvalPlayer{0};
    mutable float lastEvalVal{0.0f};
    mutable int lastEvalScaled{0};
    static constexpr int MAX_DEPTH = 64;
    static constexpr std::size_t TT_SIZE = 1u << 18;
    mutable std::vector<TTEntry> transposition; // fixed-size TT
    std::vector<std::array<int, 2>> killers;    // killer moves per depth
    std::vector<int> history;                   // history scores indexed by move id
    mutable Zobrist zobrist;
    mutable int zobristCells{0};
    FeatureExtractor extractor;
    GNNModel model;
};
