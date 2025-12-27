#include "core/Board.hpp"
#include "core/GameState.hpp"
#include "core/MoveStrategy.hpp"
#include "core/Player.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    const std::string modelPath = "../scripts/models/hex_value_ts.pt"; // adjust if you run from a different cwd
    Board board;

    // Default: human (X) vs heuristic AI (O); optional GNN AI if selected.
    char modeChoice = 'h';
    std::cout << "Jugar contra IA heuristica (h) o IA GNN (g)? [h]: ";
    if (!(std::cin >> modeChoice)) {
        modeChoice = 'h';
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    bool useGnnAi = (modeChoice == 'g' || modeChoice == 'G');

    HumanPlayer humanPlayer(1);
    AIPlayer heuristicAI(2, std::make_unique<NegamaxHeuristicStrategy>(3, 2000));
    // Give the GNN more search depth/time to compensate for higher evaluation cost.
    AIPlayer gnnAI(2, std::make_unique<NegamaxGnnStrategy>(20, 10000, modelPath));

    Player* playerX = &humanPlayer;
    Player* playerO = useGnnAi ? static_cast<Player*>(&gnnAI)
                               : static_cast<Player*>(&heuristicAI);
    Player* current = playerX;

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

        current = (current == playerX) ? playerO : playerX;
    }

    return 0;
}
