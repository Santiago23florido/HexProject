#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include <iostream>
#include "GameState.hpp"
#include "Board.hpp"

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
