#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "neural_network/activation/ActivationFunction.hpp"
#include "numerical/math/CompilerOptimizations.hpp"

namespace neural_network
{
    template<typename T>
    class LeakyReLU final
        : public ActivationFunction<T>
    {
        static_assert(std::is_floating_point_v<T>, "LeakyReLU requires a floating-point type");

    public:
        static constexpr T defaultNegativeSlope{ static_cast<T>(0.01) };

        explicit LeakyReLU(T negativeSlope = defaultNegativeSlope);

        T Forward(T x) const override;
        T Backward(T x) const override;
        void ForwardVector(std::span<T> output, std::span<const T> input) const override;
        void BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const override;

    private:
        T alpha;
    };

    template<typename T>
    LeakyReLU<T>::LeakyReLU(T negativeSlope)
        : alpha{ negativeSlope }
    {
        really_assert(alpha >= T{ 0 } && alpha < T{ 1 });
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED T LeakyReLU<T>::Forward(T x) const
    {
        return x > T{ 0 } ? x : alpha * x;
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED T LeakyReLU<T>::Backward(T x) const
    {
        return x > T{ 0 } ? T{ 1 } : alpha;
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED void LeakyReLU<T>::ForwardVector(std::span<T> output, std::span<const T> input) const
    {
        really_assert(output.size() == input.size());

        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = Forward(input[i]);
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED void LeakyReLU<T>::BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const
    {
        really_assert(result.size() == preActivation.size() && result.size() == output.size() && result.size() == outputGradient.size());

        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = outputGradient[i] * Backward(preActivation[i]);
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class LeakyReLU<float>;
#endif
}
