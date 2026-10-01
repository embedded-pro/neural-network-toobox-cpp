#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "neural_network/losses/Loss.hpp"
#include "numerical/math/CompilerOptimizations.hpp"
#include "numerical/regularization/Regularization.hpp"

namespace neural_network
{
    template<typename T, std::size_t NumberOfFeatures>
    class MeanAbsoluteError
        : public Loss<T, NumberOfFeatures>
    {
    public:
        using Vector = typename Loss<T, NumberOfFeatures>::Vector;

        MeanAbsoluteError(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm);

        T Cost(const Vector& predictions) override;
        Vector Gradient(const Vector& predictions) override;

    private:
        static constexpr T inverseSize{ T{ 1 } / static_cast<T>(NumberOfFeatures) };

        Vector target;
        regularization::Regularization<T, NumberOfFeatures>& regularization;
    };

    template<typename T, std::size_t NumberOfFeatures>
    MeanAbsoluteError<T, NumberOfFeatures>::MeanAbsoluteError(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm)
        : target{ expectedTarget }
        , regularization{ regularizationTerm }
    {}

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED T MeanAbsoluteError<T, NumberOfFeatures>::Cost(const Vector& predictions)
    {
        T sum{ 0 };

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
        {
            const T difference{ predictions[i] - target[i] };
            sum += difference < T{ 0 } ? -difference : difference;
        }

        return sum * inverseSize + regularization.Calculate(predictions);
    }

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED typename MeanAbsoluteError<T, NumberOfFeatures>::Vector MeanAbsoluteError<T, NumberOfFeatures>::Gradient(const Vector& predictions)
    {
        const Vector regularizationGradient{ regularization.Gradient(predictions) };
        Vector gradient{};

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
        {
            const T difference{ predictions[i] - target[i] };
            const T sign{ difference > T{ 0 } ? T{ 1 } : (difference < T{ 0 } ? T{ -1 } : T{ 0 }) };
            gradient[i] = sign * inverseSize + regularizationGradient[i];
        }

        return gradient;
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class MeanAbsoluteError<float, 4>;
#endif
}
