#pragma once
#include "Board.hpp"
#include "Cube.hpp"
#include <vector>
#include <iostream>

class GameState {
private:
    int N;
    std::vector<std::vector<int>> Hex;
    int Player; //1 = Player 1, 2 = Player 2
public:
    GameState(int n =7);
    GameState(const Board& b, int player);
    std::vector<int> GetAvailableMoves() const;
    std::vector<int> LinearBoard() const; // Get linear board representation index = r * N + c
    std::vector<Cube> ToCubeCoordinates() const; //Hex conversion to cube coordinates
    int IsTerminal() const;//1 = terminal,0 = not terminal
    int Winner() const; //0 = no winner, 1 = player 1 wins, 2 = player 2 wins
};