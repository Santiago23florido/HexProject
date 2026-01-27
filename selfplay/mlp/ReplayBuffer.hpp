#pragma once

#include <array>
#include <cstddef>
#include <random>
#include <vector>

/**
 *  Replay sample containing value features and a target.
 */
struct ReplaySample {
    std::array<float, 7> features{};
    float target{0.0f};
};

/**
 * Fixed-capacity replay buffer with uniform sampling.
 *
 * Owns stored samples and overwrites the oldest when full.
 */
class ReplayBuffer {
public:
    /// Creates a replay buffer with the given capacity (number of samples).
    explicit ReplayBuffer(std::size_t capacity);

    /// Adds a sample, overwriting the oldest when the buffer is full.
    void add(const ReplaySample& sample);
    /// Adds a batch of samples in order.
    void addBatch(const std::vector<ReplaySample>& samples);
    /// Clears all stored samples.
    void clear();

    /// Returns the current number of stored samples.
    std::size_t size() const;
    /// Returns the maximum capacity of the buffer.
    std::size_t capacity() const { return capacity_; }
    /// Returns true if a batch of batchSize can be sampled.
    bool canSample(std::size_t batchSize) const;

    /// Samples batchSize entries uniformly with replacement.
    std::vector<ReplaySample> sample(std::size_t batchSize, std::mt19937& rng) const;
    /// Returns a read-only view of the underlying storage.
    const std::vector<ReplaySample>& data() const { return buffer_; }

private:
    std::size_t capacity_{0};
    std::size_t next_{0};
    bool full_{false};
    std::vector<ReplaySample> buffer_;
};
