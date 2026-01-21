#include "core/Board.hpp"
#include <iostream>
#include <stdexcept>
#include <type_traits>
using namespace std;

namespace {
int ValidateBoardSize(int n) {
    if (n <= 0) {
        throw std::invalid_argument("Board size must be positive");
    }
    return n;
}

template <typename T, typename Enable = void>
struct RangeTraits;

template <typename T>
struct RangeTraits<T, std::enable_if_t<std::is_signed<T>::value>> {
    static bool InRange(T value, T minValue, T maxValue) {
        return value >= minValue && value < maxValue;
    }
};

template <typename T>
struct RangeTraits<T, std::enable_if_t<std::is_unsigned<T>::value>> {
    static bool InRange(T value, T minValue, T maxValue) {
        return value >= minValue && value < maxValue;
    }
};

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
bool InRange(T value, T minValue, T maxValue) {
    return RangeTraits<T>::InRange(value, minValue, maxValue);
}
} // namespace

Board::Board(int n) : N(ValidateBoardSize(n)), board(N, std::vector<int>(N, 0)) {}
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
    if (!InRange(r, 0, N) || !InRange(c, 0, N)) {
        throw std::out_of_range("Board::place row/column out of range");
    }

    auto row = board.begin() + r;
    auto cell = row->begin() + c;

    if (*cell != 0) return false;

    *cell = player;
    return true;
}

// Linear index variant: idx = r * N + c
bool Board::place(int idx, int player) {
    if (!InRange(idx, 0, N * N)) {
        throw std::out_of_range("Board::place index out of range");
    }
    const int r = idx / N;
    const int c = idx % N;
    return place(r, c, player);
}
