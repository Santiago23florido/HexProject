#include "Board.hpp"
#include <iostream>
using namespace std;

Board::Board(int n) : N(n), board(n, std::vector<int>(n, 0)) {}
Board::Board(const Board& other) : N(other.N), board(other.board) {}

void Board::print() const {
    cout << "\nHEX BOARD " << N << "x" << N << "\n\n";

    int row_index = 0;
    for (auto row = board.begin(); row != board.end(); ++row) {

       //Desplacement of rows for hexagonal demolayout
        for (int s = 0; s < row_index; s++)
            cout << " ";

        // Columns Iteration
        auto col = row->begin();
        while (col != row->end()) {

            char symbol = '.';
            if (*col == 1) symbol = 'X';
            else if (*col == 2) symbol = 'O';

            cout << symbol;

            ++col;
            if (col != row->end())
                cout << " ";
        }

        cout << "\n";
        row_index++;
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
