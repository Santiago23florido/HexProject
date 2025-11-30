#include "GameState.hpp"
#include "Board.hpp"
#include "Cube.hpp"
#include "MoveStrategy.hpp"
#include <vector>
#include <iostream>
#include <unordered_map>
#include <algorithm>

//Random selection from the available moves list
int RandomStrategy::select(const GameState& state, int playerId) {
    std::vector<int, std::allocator<int>> moves = state.GetAvailableMoves();
    return moves[rand() % moves.size()];
}

