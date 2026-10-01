#include "simulator/model/NeuralNetwork/application/Mlp.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace simulator::model::nn
{
    namespace
    {
        float Sigmoid(float x)
        {
            if (x >= 0.0f)
                return 1.0f / (1.0f + std::exp(-x));

            const auto e{ std::exp(x) };
            return e / (1.0f + e);
        }

        void FillNormal(std::span<float> values, float standardDeviation, std::mt19937& rng)
        {
            std::normal_distribution<float> distribution{ 0.0f, standardDeviation };

            for (auto& value : values)
                value = distribution(rng);
        }
    }

    Mlp::Mlp(std::size_t inputSize, std::size_t hiddenSize, std::uint32_t seed)
        : inputSize{ inputSize }
        , hiddenSize{ hiddenSize }
        , parameters(hiddenSize * inputSize + 2 * hiddenSize + 1, 0.0f)
        , gradients(parameters.size(), 0.0f)
        , hidden(hiddenSize, 0.0f)
    {
        std::mt19937 rng{ seed };
        const std::span<float> all{ parameters };

        FillNormal(all.first(hiddenSize * inputSize), std::sqrt(1.0f / static_cast<float>(inputSize)), rng);
        FillNormal(all.subspan(OutputWeightOffset(), hiddenSize), std::sqrt(1.0f / static_cast<float>(hiddenSize)), rng);
    }

    float Mlp::Predict(std::span<const float> input)
    {
        auto output{ parameters[OutputBiasOffset()] };

        for (std::size_t row = 0; row < hiddenSize; ++row)
        {
            auto activation{ parameters[HiddenBiasOffset() + row] };

            for (std::size_t column = 0; column < inputSize; ++column)
                activation += parameters[row * inputSize + column] * input[column];

            hidden[row] = std::tanh(activation);
            output += parameters[OutputWeightOffset() + row] * hidden[row];
        }

        return Sigmoid(output);
    }

    float Mlp::Accumulate(std::span<const float> input, float target)
    {
        const auto prediction{ Predict(input) };
        const auto error{ prediction - target };
        const auto outputDelta{ 2.0f * error * prediction * (1.0f - prediction) };

        gradients[OutputBiasOffset()] += outputDelta;

        for (std::size_t row = 0; row < hiddenSize; ++row)
        {
            gradients[OutputWeightOffset() + row] += outputDelta * hidden[row];

            const auto hiddenDelta{ outputDelta * parameters[OutputWeightOffset() + row] * (1.0f - hidden[row] * hidden[row]) };
            gradients[HiddenBiasOffset() + row] += hiddenDelta;

            for (std::size_t column = 0; column < inputSize; ++column)
                gradients[row * inputSize + column] += hiddenDelta * input[column];
        }

        return error * error;
    }

    void Mlp::Step(float learningRate, std::size_t batchSize)
    {
        const auto scale{ learningRate / static_cast<float>(batchSize) };

        for (std::size_t index = 0; index < parameters.size(); ++index)
            parameters[index] -= scale * gradients[index];

        std::fill(gradients.begin(), gradients.end(), 0.0f);
    }

    std::span<float> Mlp::Parameters()
    {
        return parameters;
    }

    std::span<const float> Mlp::Gradients() const
    {
        return gradients;
    }

    std::size_t Mlp::HiddenBiasOffset() const
    {
        return hiddenSize * inputSize;
    }

    std::size_t Mlp::OutputWeightOffset() const
    {
        return HiddenBiasOffset() + hiddenSize;
    }

    std::size_t Mlp::OutputBiasOffset() const
    {
        return OutputWeightOffset() + hiddenSize;
    }
}
