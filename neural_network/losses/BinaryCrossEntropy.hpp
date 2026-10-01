#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "neural_network/losses/Loss.hpp"
#include "numerical/math/CompilerOptimizations.hpp"
#include "numerical/math/Math.hpp"
#include "numerical/regularization/Regularization.hpp"
#include <algorithm>

namespace neural_network
{
    template<typename T, std::size_t NumberOfFeatures>
    class BinaryCrossEntropy
        : public Loss<T, NumberOfFeatures>
    {
    public:
        using Vector = typename Loss<T, NumberOfFeatures>::Vector;

        static constexpr T epsilon{ static_cast<T>(1e-7) };

        BinaryCrossEntropy(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm);

        T Cost(const Vector& predictions) override;
        Vector Gradient(const Vector& predictions) override;

    private:
        static constexpr T inverseSize{ T{ 1 } / static_cast<T>(NumberOfFeatures) };

        static T ClampProbability(T probability);

        Vector target;
        regularization::Regularization<T, NumberOfFeatures>& regularization;
    };

    template<typename T, std::size_t NumberOfFeatures>
    BinaryCrossEntropy<T, NumberOfFeatures>::BinaryCrossEntropy(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm)
        : target{ expectedTarget }
        , regularization{ regularizationTerm }
    {}

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED T BinaryCrossEntropy<T, NumberOfFeatures>::Cost(const Vector& predictions)
    {
        T sum{ 0 };

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
        {
            const T probability{ ClampProbability(predictions[i]) };
            sum -= target[i] * math::Log(probability) + (T{ 1 } - target[i]) * math::Log(T{ 1 } - probability);
        }

        return sum * inverseSize + regularization.Calculate(predictions);
    }

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED typename BinaryCrossEntropy<T, NumberOfFeatures>::Vector BinaryCrossEntropy<T, NumberOfFeatures>::Gradient(const Vector& predictions)
    {
        const Vector regularizationGradient{ regularization.Gradient(predictions) };
        Vector gradient{};

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
        {
            const T probability{ ClampProbability(predictions[i]) };
            gradient[i] = (probability - target[i]) / (probability * (T{ 1 } - probability)) * inverseSize + regularizationGradient[i];
        }

        return gradient;
    }

    template<typename T, std::size_t NumberOfFeatures>
    T BinaryCrossEntropy<T, NumberOfFeatures>::ClampProbability(T probability)
    {
        return std::clamp(probability, epsilon, T{ 1 } - epsilon);
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class BinaryCrossEntropy<float, 2>;
#endif
}
