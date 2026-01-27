#include "Serializer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

// Implements JSON Lines serialization for Sample data with one object per line.

bool Serializer::writeJsonl(const std::vector<Sample>& samples, const std::string& path, bool append) {
    // Throws std::runtime_error on open or write failure.
    std::ios_base::openmode mode = append ? std::ios::app : std::ios::trunc;
    std::ofstream out(path, mode);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    for (const auto& s : samples) {
        out << "{";
        out << "\"N\":" << s.N << ",";
        out << "\"to_move\":" << s.toMove << ",";
        out << "\"result\":" << s.result << ",";
        out << "\"moves_to_end\":" << s.movesToEnd << ",";
        out << "\"board\":[";
        for (size_t i = 0; i < s.board.size(); ++i) {
            out << s.board[i];
            if (i + 1 < s.board.size()) out << ",";
        }
        out << "]";
        out << "}\n";
    }
    if (!out.good()) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
    return true;
}
