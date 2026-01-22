#include "ReplayBuffer.hpp"

ReplayBuffer::ReplayBuffer(std::size_t capacity)
    : capacity_(capacity) {
    buffer_.reserve(capacity_);
}

void ReplayBuffer::add(const ReplaySample& sample) {
    if (capacity_ == 0) return;

    if (buffer_.size() < capacity_) {
        buffer_.push_back(sample);
        if (buffer_.size() == capacity_) {
            full_ = true;
            next_ = 0;
        }
        return;
    }

    buffer_[next_] = sample;
    next_ = (next_ + 1) % capacity_;
    full_ = true;
}

void ReplayBuffer::addBatch(const std::vector<ReplaySample>& samples) {
    for (const auto& s : samples) {
        add(s);
    }
}

void ReplayBuffer::clear() {
    buffer_.clear();
    next_ = 0;
    full_ = false;
}

std::size_t ReplayBuffer::size() const {
    return buffer_.size();
}

bool ReplayBuffer::canSample(std::size_t batchSize) const {
    return buffer_.size() >= batchSize && batchSize > 0;
}

std::vector<ReplaySample> ReplayBuffer::sample(std::size_t batchSize, std::mt19937& rng) const {
    std::vector<ReplaySample> out;
    if (batchSize == 0 || buffer_.empty()) return out;
    out.reserve(batchSize);
    std::uniform_int_distribution<std::size_t> dist(0, buffer_.size() - 1);
    for (std::size_t i = 0; i < batchSize; ++i) {
        out.push_back(buffer_[dist(rng)]);
    }
    return out;
}
