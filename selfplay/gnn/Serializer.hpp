#pragma once

#include <string>
#include <vector>

#include "gnn/DataCollector.hpp"

/**
 * Serializes collected samples to JSON Lines files.
 */
class Serializer {
public:
    /// Writes samples to a JSONL file at path, appending if requested.
    static bool writeJsonl(const std::vector<Sample>& samples, const std::string& path, bool append = false);
};
