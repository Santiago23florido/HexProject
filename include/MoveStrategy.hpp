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
    int simulationsPerMove;
public:
    MonteCarloStrategy(int sims);
    int select(const GameState& state, int playerId) override; 
private:
    int simulate(GameState state, int playerId); 
};

