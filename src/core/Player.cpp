#include "core/Player.hpp"
#include "core/MoveStrategy.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <stdexcept>

// Clone to a simple random strategy when a strategy exists
static std::unique_ptr<IMoveStrategy> CloneStrategy(const std::unique_ptr<IMoveStrategy>& s) {
    return s ? std::make_unique<RandomStrategy>() : nullptr;
}

// Shared human input handler
static int PromptHumanMove(const GameState& state) {
    const auto moves = state.GetAvailableMoves();
    if (moves.empty()) {
        return -1;
    }

    const int size = static_cast<int>(state.LinearBoard().size());
    const int n = static_cast<int>(std::sqrt(size));

    while (true) {
        int r = 0, c = 0;
        std::cout << "Enter row and column: ";
        std::cin >> r >> c;
        if (!std::cin) {
            throw std::runtime_error("Failed to read move input");
        }

        if (r >= 0 && r < n && c >= 0 && c < n) {
            const int idx = r * n + c;
            if (std::find(moves.begin(), moves.end(), idx) != moves.end()) {
                return idx;
            }
        }
        std::cout << "Invalid move, try again.\n";
    }
}

HumanPlayer::HumanPlayer(int id) : PlayerId(id) {}

int HumanPlayer::Id() const {
    return PlayerId;
}

// Ask for a valid linear move index
int HumanPlayer::ChooseMove(const GameState& state) {
    return PromptHumanMove(state);
}

AIPlayer::AIPlayer(int id, std::unique_ptr<IMoveStrategy> s)
    : playerId(id), strategy(std::move(s)) {}

AIPlayer::AIPlayer(int id)
    : playerId(id), strategy(std::make_unique<RandomStrategy>()) {}

int AIPlayer::Id() const {
    return playerId;
}

// Delegate to configured strategy
int AIPlayer::ChooseMove(const GameState& state) {
    return strategy->select(state, playerId);
}

IMoveStrategy* AIPlayer::Strategy() {
    return strategy.get();
}

const IMoveStrategy* AIPlayer::Strategy() const {
    return strategy.get();
}

HybridPlayer::HybridPlayer(int id)
    : playerId(id) {}

HybridPlayer::HybridPlayer(int id, std::unique_ptr<IMoveStrategy> s)
    : playerId(id), strategy(std::move(s)) {}

HybridPlayer::HybridPlayer(const HumanPlayer& other)
    : playerId(other.Id()) {}

HybridPlayer::HybridPlayer(const AIPlayer& other)
    : playerId(other.Id()), strategy(std::make_unique<RandomStrategy>()) {}

HybridPlayer::HybridPlayer(const HybridPlayer& other)
    : playerId(other.Id()),
      strategy(CloneStrategy(other.strategy)) {}

HybridPlayer& HybridPlayer::operator=(const HybridPlayer& other) {
    if (this != &other) {
        playerId = other.Id();
        strategy = CloneStrategy(other.strategy);
    }
    return *this;
}

int HybridPlayer::Id() const {
    return playerId;
}

// Uses AI when a strategy is set, otherwise behaves like a human player
int HybridPlayer::ChooseMove(const GameState& state) {
    if (strategy) {
        return strategy->select(state, playerId);
    }
    return PromptHumanMove(state);
}
