#include "Serializer.hpp"

#include <fstream>
#include <sstream>

bool Serializer::writeJsonl(const std::vector<Sample>& samples, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    for (const auto& s : samples) {
        out << "{";
        out << "\"N\":" << s.N << ",";
        out << "\"to_move\":" << s.toMove << ",";
        out << "\"result\":" << s.result << ",";
        out << "\"board\":[";
        for (size_t i = 0; i < s.board.size(); ++i) {
            out << s.board[i];
            if (i + 1 < s.board.size()) out << ",";
        }
        out << "]";
        out << "}\n";
    }
    return true;
}
