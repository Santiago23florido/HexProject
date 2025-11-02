#include "Board.hpp"
#include "GameState.hpp"
#include <iostream>

int main() {
    Board board;
    int player = 1;

    while (true) {
        board.print();
        std::cout << "\nPlayer " << (player == 1 ? "X" : "O") << " turn\n";
        std::cout << "Enter row and column: ";

        int r, c;
        std::cin >> r >> c;

        if (!board.place(r, c, player)) {
            std::cout << "Invalid move, try again.\n";
            continue;
        }

        GameState state(board, player);
        int w = state.Winner();
        if (w == 1) {
            board.print();
            std::cout << "\nPlayer X wins!\n";
            break;
        }
        if (w == 2) {
            board.print();
            std::cout << "\nPlayer O wins!\n";
            break;
        }

        player = (player == 1 ? 2 : 1);
    }

    return 0;
}
