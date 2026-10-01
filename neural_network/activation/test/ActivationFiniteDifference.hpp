#pragma once

#include "neural_network/activation/ActivationFunction.hpp"
#include <array>
#include <cstddef>

namespace neural_network::test_support
{
    inline constexpr float finiteDifferenceStep{ 1e-3f };

    inline float CentralDifference(const ActivationFunction<float>& activation, float x)
    {
        return (activation.Forward(x + finiteDifferenceStep) - activation.Forward(x - finiteDifferenceStep)) / (2.0f * finiteDifferenceStep);
    }

    template<std::size_t Size>
    float ProjectedOutput(const ActivationFunction<float>& activation, const std::array<float, Size>& input, const std::array<float, Size>& upstream)
    {
        std::array<float, Size> output{};
        activation.ForwardVector(output, input);

        float sum{ 0.0f };
        for (std::size_t i = 0; i < Size; ++i)
            sum += upstream[i] * output[i];

        return sum;
    }

    template<std::size_t Size>
    std::array<float, Size> CentralDifferenceGradient(const ActivationFunction<float>& activation, const std::array<float, Size>& input, const std::array<float, Size>& upstream)
    {
        std::array<float, Size> gradient{};

        for (std::size_t i = 0; i < Size; ++i)
        {
            auto plus{ input };
            auto minus{ input };
            plus[i] += finiteDifferenceStep;
            minus[i] -= finiteDifferenceStep;
            gradient[i] = (ProjectedOutput(activation, plus, upstream) - ProjectedOutput(activation, minus, upstream)) / (2.0f * finiteDifferenceStep);
        }

        return gradient;
    }

    template<std::size_t Size>
    std::array<float, Size> AnalyticGradient(const ActivationFunction<float>& activation, const std::array<float, Size>& input, const std::array<float, Size>& upstream)
    {
        std::array<float, Size> output{};
        activation.ForwardVector(output, input);

        std::array<float, Size> result{};
        activation.BackwardVector(result, input, output, upstream);

        return result;
    }
}
