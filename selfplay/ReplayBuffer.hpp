#pragma once

#include <array>
#include <cstddef>
#include <random>
#include <vector>

struct ReplaySample {
    std::array<float, 7> features{};
    float target{0.0f};
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(std::size_t capacity);

    void add(const ReplaySample& sample);
    void addBatch(const std::vector<ReplaySample>& samples);
    void clear();

    std::size_t size() const;
    std::size_t capacity() const { return capacity_; }
    bool canSample(std::size_t batchSize) const;

    std::vector<ReplaySample> sample(std::size_t batchSize, std::mt19937& rng) const;
    const std::vector<ReplaySample>& data() const { return buffer_; }

private:
    std::size_t capacity_{0};
    std::size_t next_{0};
    bool full_{false};
    std::vector<ReplaySample> buffer_;
};
