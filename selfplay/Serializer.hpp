#pragma once

#include <string>
#include <vector>

#include "DataCollector.hpp"

class Serializer {
public:
    static bool writeJsonl(const std::vector<Sample>& samples, const std::string& path);
};
