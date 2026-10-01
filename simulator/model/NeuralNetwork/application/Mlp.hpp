#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace simulator::model::nn
{
    class Mlp
    {
    public:
        Mlp(std::size_t inputSize, std::size_t hiddenSize, std::uint32_t seed);

        [[nodiscard]] float Predict(std::span<const float> input);
        float Accumulate(std::span<const float> input, float target);
        void Step(float learningRate, std::size_t batchSize);

        [[nodiscard]] std::span<float> Parameters();
        [[nodiscard]] std::span<const float> Gradients() const;

    private:
        [[nodiscard]] std::size_t HiddenBiasOffset() const;
        [[nodiscard]] std::size_t OutputWeightOffset() const;
        [[nodiscard]] std::size_t OutputBiasOffset() const;

        std::size_t inputSize;
        std::size_t hiddenSize;
        std::vector<float> parameters;
        std::vector<float> gradients;
        std::vector<float> hidden;
    };
}
