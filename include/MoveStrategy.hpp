#pragma once
#include <vector>
#include <iostream>
#include "GameState.hpp"

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
