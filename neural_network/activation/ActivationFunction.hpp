#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "infra/util/ReallyAssert.hpp"
#include "numerical/math/CompilerOptimizations.hpp"
#include <cstddef>
#include <span>
#include <type_traits>

namespace neural_network
{
    template<typename T>
    class ActivationFunction
    {
        static_assert(std::is_floating_point_v<T>, "ActivationFunction requires a floating-point type");

    public:
        virtual ~ActivationFunction() = default;

        virtual T Forward(T x) const = 0;
        virtual T Backward(T x) const = 0;

        virtual void ForwardVector(std::span<T> output, std::span<const T> input) const;
        virtual void BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const;

    protected:
        ActivationFunction() = default;
        ActivationFunction(const ActivationFunction&) = default;
        ActivationFunction& operator=(const ActivationFunction&) = default;
    };

    template<typename T>
    OPTIMIZE_FOR_SPEED void ActivationFunction<T>::ForwardVector(std::span<T> output, std::span<const T> input) const
    {
        really_assert(output.size() == input.size());

        for (std::size_t i = 0; i < output.size(); ++i)
            output[i] = Forward(input[i]);
    }

    template<typename T>
    OPTIMIZE_FOR_SPEED void ActivationFunction<T>::BackwardVector(std::span<T> result, std::span<const T> preActivation, std::span<const T> output, std::span<const T> outputGradient) const
    {
        really_assert(result.size() == preActivation.size() && result.size() == output.size() && result.size() == outputGradient.size());

        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = outputGradient[i] * Backward(preActivation[i]);
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class ActivationFunction<float>;
#endif
}
