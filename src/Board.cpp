#include "Board.hpp"
#include <iostream>
using namespace std;

Board::Board(int n) : N(n), board(n, std::vector<int>(n, 0)) {}
Board::Board(const Board& other) : N(other.N), board(other.board) {}

void Board::print() const {
    std::cout << "\nHEX BOARD " << N << "x" << N << "\n\n";

    for (int r = 0; r < N; ++r) {

        // Layout tipo odd-r: filas pares sin indent, impares con desplazamiento.
        int indent = (r % 2 == 0) ? 0 : 1;
        for (int s = 0; s < indent; ++s)
            std::cout << " ";

        for (int c = 0; c < N; ++c) {
            char symbol = '.';
            if (board[r][c] == 1) symbol = 'X';
            else if (board[r][c] == 2) symbol = 'O';

            std::cout << symbol;
            if (c + 1 < N)
                std::cout << " ";
        }
        std::cout << "\n";
    }
}

bool Board::place(int r, int c, int player) {
    // User input placement validation
    if (r < 0 || r >= N || c < 0 || c >= N) return false;

    auto row = board.begin() + r;
    auto cell = row->begin() + c;

    if (*cell != 0) return false;

    *cell = player;
    return true;
}

// Linear index variant: idx = r * N + c
bool Board::place(int idx, int player) {
    if (idx < 0 || idx >= N * N) return false;
    const int r = idx / N;
    const int c = idx % N;
    return place(r, c, player);
}
