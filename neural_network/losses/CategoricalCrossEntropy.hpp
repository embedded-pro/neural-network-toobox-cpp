#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "neural_network/losses/Loss.hpp"
#include "numerical/math/CompilerOptimizations.hpp"
#include "numerical/math/Math.hpp"
#include "numerical/regularization/Regularization.hpp"

namespace neural_network
{
    template<typename T, std::size_t NumberOfFeatures>
    class CategoricalCrossEntropy
        : public Loss<T, NumberOfFeatures>
    {
    public:
        using Vector = typename Loss<T, NumberOfFeatures>::Vector;

        CategoricalCrossEntropy(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm);

        T Cost(const Vector& logits) override;
        Vector Gradient(const Vector& logits) override;

    private:
        static T MaxLogit(const Vector& logits);
        static T ShiftedLogSumExp(const Vector& logits, T maxLogit);
        T TargetSum() const;

        Vector target;
        regularization::Regularization<T, NumberOfFeatures>& regularization;
    };

    template<typename T, std::size_t NumberOfFeatures>
    CategoricalCrossEntropy<T, NumberOfFeatures>::CategoricalCrossEntropy(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm)
        : target{ expectedTarget }
        , regularization{ regularizationTerm }
    {}

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED T CategoricalCrossEntropy<T, NumberOfFeatures>::Cost(const Vector& logits)
    {
        const T maxLogit{ MaxLogit(logits) };
        const T shiftedLogSumExp{ ShiftedLogSumExp(logits, maxLogit) };

        T weightedShifted{ 0 };
        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
            weightedShifted += target[i] * (logits[i] - maxLogit);

        return TargetSum() * shiftedLogSumExp - weightedShifted + regularization.Calculate(logits);
    }

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED typename CategoricalCrossEntropy<T, NumberOfFeatures>::Vector CategoricalCrossEntropy<T, NumberOfFeatures>::Gradient(const Vector& logits)
    {
        const Vector regularizationGradient{ regularization.Gradient(logits) };
        const T maxLogit{ MaxLogit(logits) };
        const T targetSum{ TargetSum() };

        Vector gradient{};
        T sum{ 0 };
        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
        {
            gradient[i] = math::Exp(logits[i] - maxLogit);
            sum += gradient[i];
        }

        const T scale{ targetSum / sum };
        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
            gradient[i] = gradient[i] * scale - target[i] + regularizationGradient[i];

        return gradient;
    }

    template<typename T, std::size_t NumberOfFeatures>
    T CategoricalCrossEntropy<T, NumberOfFeatures>::MaxLogit(const Vector& logits)
    {
        T maxLogit{ logits[0] };
        for (std::size_t i = 1; i < NumberOfFeatures; ++i)
            if (logits[i] > maxLogit)
                maxLogit = logits[i];

        return maxLogit;
    }

    template<typename T, std::size_t NumberOfFeatures>
    T CategoricalCrossEntropy<T, NumberOfFeatures>::ShiftedLogSumExp(const Vector& logits, T maxLogit)
    {
        T sum{ 0 };
        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
            sum += math::Exp(logits[i] - maxLogit);

        return math::Log(sum);
    }

    template<typename T, std::size_t NumberOfFeatures>
    T CategoricalCrossEntropy<T, NumberOfFeatures>::TargetSum() const
    {
        T sum{ 0 };
        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
            sum += target[i];

        return sum;
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class CategoricalCrossEntropy<float, 2>;
#endif
}
