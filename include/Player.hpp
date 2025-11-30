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
    AIPlayer(int id, std::unique_ptr<IMoveStrategy> s);
    int ChooseMove(const GameState& state) override;
    int Id() const override;
};
