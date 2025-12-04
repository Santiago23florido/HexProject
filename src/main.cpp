#include "Board.hpp"
#include "GameState.hpp"
#include "MoveStrategy.hpp"
#include "Player.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    Board board;
    const std::string modelPath = "../scripts/models/hex_value_ts.pt"; // adjust if you run from a different cwd
    const bool heuristicOnly = true;
    AIPlayer player1(1, std::make_unique<NegamaxStrategy>(3, 2000, modelPath, heuristicOnly));
    HumanPlayer player2(2);
    Player* current = &player1;

    while (true) {
        board.print();
        std::cout << "\nPlayer " << (current->Id() == 1 ? "X" : "O") << " turn\n";
        GameState state(board, current->Id());
        int moveIdx = current->ChooseMove(state);
        if (!board.place(moveIdx, current->Id())) {
            std::cout << "Invalid move, try again.\n";
            continue;
        }
        state.Update(board, current->Id());
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

        current = (current == &player1) ? static_cast<Player*>(&player2) : static_cast<Player*>(&player1);
    }

    return 0;
}
