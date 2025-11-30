#pragma once
#include <memory>
#include "GameState.hpp"
#include "MoveStrategy.hpp"

// Player base interface
class Player {
    public:
        virtual int ChooseMove(const GameState& state)=0;
        virtual int Id() const = 0;
        virtual ~Player() = default;
};

class HumanPlayer: public Player {
    int PlayerId; // immutable identity
    public:
    HumanPlayer(int id);
    int ChooseMove(const GameState& state) override;
    int Id() const override;
};

class AIPlayer : public Player {
    int playerId;
    std::unique_ptr<IMoveStrategy> strategy;
public:
    AIPlayer(int id); // defaults to RandomStrategy
    AIPlayer(int id, std::unique_ptr<IMoveStrategy> s);
    int ChooseMove(const GameState& state) override;
    int Id() const override;
};


// Can act as a human (no strategy) or AI (with strategy)
class HybridPlayer : public Player {
    int playerId;
    std::unique_ptr<IMoveStrategy> strategy;
public:
    HybridPlayer(int id); // human by default
    HybridPlayer(int id, std::unique_ptr<IMoveStrategy> s); // AI when strategy is provided
    HybridPlayer(const HumanPlayer& other); // copy identity, stay human
    HybridPlayer(const AIPlayer& other);    // copy identity, become AI (RandomStrategy)
    HybridPlayer(const HybridPlayer& other); // copy identity, keep mode (human/AI)
    HybridPlayer& operator=(const HybridPlayer& other);
    HybridPlayer(HybridPlayer&&) noexcept = default;
    HybridPlayer& operator=(HybridPlayer&&) noexcept = default;
    int ChooseMove(const GameState& state) override;
    int Id() const override;
};
