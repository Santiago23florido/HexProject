#include "Player.hpp"
#include "MoveStrategy.hpp"
#include <algorithm>
#include <iostream>
#include <math.h>

HumanPlayer::HumanPlayer(int id) : PlayerId(id) {}

int HumanPlayer::Id() const {
    return PlayerId;
}

// Ask for a valid linear move index
int HumanPlayer::ChooseMove(const GameState& state) {
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

        if (r >= 0 && r < n && c >= 0 && c < n) {
            const int idx = r * n + c;
            if (std::find(moves.begin(), moves.end(), idx) != moves.end()) {
                return idx;  
            }
        }
        std::cout << "Invalid move, try again.\n";
    }
}

AIPlayer::AIPlayer(int id, std::unique_ptr<IMoveStrategy> s)
    : playerId(id), strategy(std::move(s)) {}

int AIPlayer::Id() const {
    return playerId;
}

// Delegate to configured strategy
int AIPlayer::ChooseMove(const GameState& state) {
    return strategy->select(state, playerId);
}
