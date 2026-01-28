#include "core/Cube.hpp"

// Implements cube coordinate utilities used for hex-grid neighbor calculations.
Cube::Cube() : x(0), y(0), z(0) {}
Cube::Cube(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
long long Cube::key() const{
    return ((long long)x << 40) ^ ((long long)y << 20) ^ (long long)z;
}
Cube Cube::operator+(const Cube& other) const{
    return Cube(x + other.x, y + other.y, z + other.z);
}
