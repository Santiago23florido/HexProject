#pragma once
#include <vector>
#include <iostream>

/**
 * Hex board storage and placement utilities.
 *
 * Owns an N x N grid; cells store 0 (empty), 1 or 2 (player).
 */
class Board {
public:
    int N;
    std::vector<std::vector<int>> board; //0 = free cell, 1 = Player 1 , 2 Player 2
    /// Creates an N x N board (N > 0).
    Board(int n = 7);    
    /// Copy-constructs a board from another.
    Board(const Board& other);
    /// Prints the board state to stdout.
    void print() const;  
    /// Places a stone at (r,c); returns false if occupied.
    bool place(int r, int c, int player); 
    /// Places a stone at linear index idx = r * N + c; returns false if occupied.
    bool place(int idx, int player); // linear index helper
};
