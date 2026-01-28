#pragma once
#include "core/Board.hpp"
#include "core/Cube.hpp"
#include <vector>
#include <iostream>

/**
 *Snapshot of a game position with current player.
 *
 * Owns a copy of the board state and exposes move generation and win checks.
 */
class GameState {
private:
    int N;
    std::vector<std::vector<int>> Hex;
    int Player; //1 = Player 1, 2 = Player 2
public:
    /// Creates an empty N x N game state.
    GameState(int n =7);
    /// Creates a state from a board and current player id.
    GameState(const Board& b, int player);
    /// Copy-constructs a game state.
    GameState(const GameState& other);
    /// Returns linear indices of empty cells.
    std::vector<int> GetAvailableMoves() const;
    /// Returns the board as a linear array (idx = r * N + c).
    std::vector<int> LinearBoard() const; // Get linear board representation index = r * N + c
    /// Converts the board to cube coordinates (size N*N).
    std::vector<Cube> ToCubeCoordinates() const; //Hex conversion to cube coordinates
    /// Returns 1 if terminal, 0 otherwise.
    int IsTerminal() const;//1 = terminal,0 = not terminal
    /// Returns the winner id (0 = none, 1 = player 1, 2 = player 2).
    int Winner() const; //0 = no winner, 1 = player 1 wins, 2 = player 2 wins
    /// Updates the state from a board and current player.
    void Update(const Board& b, int player);
};
