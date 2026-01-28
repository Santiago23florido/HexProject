#pragma once
#include "core/Board.hpp"
#include <vector>
#include <iostream>

/**
 *Cube coordinates for hex-grid neighbor arithmetic.
 */
class Cube {
    public:
    int x;
    int y;
    int z;
    /// Creates cube coordinates (x, y, z).
    Cube(int x_,int y_, int z_);
    /// Returns a hashable key for the coordinate.
    long long key() const;
    /// Creates a zero coordinate (0,0,0).
    Cube();
    /// Adds two cube coordinates.
    Cube operator+(const Cube& other) const; // Adition cube coordinates operator
};//Cube coordinates structure
