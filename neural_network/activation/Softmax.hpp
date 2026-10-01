#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "neural_network/activation/ActivationFunction.hpp"
#include "numerical/math/CompilerOptimizations.hpp"
#include "numerical/math/Math.hpp"

namespace neural_network
{
    template<typename T>
    class Softmax final
        : public ActivationFunction<T>
    {
        static_assert(std::is_floating_point_v<T>, "Softmax requires a floating-point type");

    public:
        T Forward(T x) const override;
        T Backward(T x) const override;
        void ForwardVector(std::span<T> output, std::span<const T> input) const override;
        void BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const override;
    };

    template<typename T>
    OPTIMIZE_FOR_SPEED T Softmax<T>::Forward(T) const
    {
        return T{ 1 };
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED T Softmax<T>::Backward(T) const
    {
        return T{ 0 };
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED void Softmax<T>::ForwardVector(std::span<T> output, std::span<const T> input) const
    {
        really_assert(!input.empty() && output.size() == input.size());

        T maxValue{ input[0] };
        for (std::size_t i = 1; i < input.size(); ++i)
            if (input[i] > maxValue)
                maxValue = input[i];

        T sum{ 0 };
        for (std::size_t i = 0; i < output.size(); ++i)
        {
            output[i] = math::Exp(input[i] - maxValue);
            sum += output[i];
        }

        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] /= sum;
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED void Softmax<T>::BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const
    {
        really_assert(result.size() == preActivation.size() && result.size() == output.size() && result.size() == outputGradient.size());

        T dot{ 0 };
        for (std::size_t i = 0; i < result.size(); ++i)
            dot += outputGradient[i] * output[i];

        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = output[i] * (outputGradient[i] - dot);
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class Softmax<float>;
#endif
}
