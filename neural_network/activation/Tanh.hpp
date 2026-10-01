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
    class Tanh final
        : public ActivationFunction<T>
    {
        static_assert(std::is_floating_point_v<T>, "Tanh requires a floating-point type");

    public:
        T Forward(T x) const override;
        T Backward(T x) const override;
        void ForwardVector(std::span<T> output, std::span<const T> input) const override;
        void BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const override;
    };

    template<typename T>
    OPTIMIZE_FOR_SPEED T Tanh<T>::Forward(T x) const
    {
        return math::Tanh(x);
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED T Tanh<T>::Backward(T x) const
    {
        const T y{ Forward(x) };
        return T{ 1 } - y * y;
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED void Tanh<T>::ForwardVector(std::span<T> output, std::span<const T> input) const
    {
        really_assert(output.size() == input.size());

        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = Forward(input[i]);
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED void Tanh<T>::BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const
    {
        really_assert(result.size() == preActivation.size() && result.size() == output.size() && result.size() == outputGradient.size());

        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = outputGradient[i] * (T{ 1 } - output[i] * output[i]);
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class Tanh<float>;
#endif
}
