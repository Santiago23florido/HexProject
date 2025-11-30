#include "Board.hpp"
#include "GameState.hpp"
#include "Player.hpp"
#include <iostream>

HumanPlayer Player1 = HumanPlayer(1);
HumanPlayer Player2 = HumanPlayer(2);

int main() {
    Board board;
    HumanPlayer Player = Player1;

    while (true) {
        board.print();
        std::cout << "\nPlayer " << (Player.Id() == 1 ? "X" : "O") << " turn\n";
        std::cout << "Enter row and column: ";
        GameState state(board, Player.Id());
        int Move = Player.ChooseMove(state);
        board.place(Move,Player.Id());
        state.Update(board,Player.Id());
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

        Player = (Player.Id() == Player1.Id() ? Player2 : Player1);
    }

    return 0;
}
