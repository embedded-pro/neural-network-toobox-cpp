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
    class MeanSquaredError
        : public Loss<T, NumberOfFeatures>
    {
    public:
        using Vector = typename Loss<T, NumberOfFeatures>::Vector;

        MeanSquaredError(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm);

        T Cost(const Vector& predictions) override;
        Vector Gradient(const Vector& predictions) override;

    private:
        static constexpr T inverseSize{ T{ 1 } / static_cast<T>(NumberOfFeatures) };

        Vector target;
        regularization::Regularization<T, NumberOfFeatures>& regularization;
    };

    template<typename T, std::size_t NumberOfFeatures>
    MeanSquaredError<T, NumberOfFeatures>::MeanSquaredError(const Vector& expectedTarget, regularization::Regularization<T, NumberOfFeatures>& regularizationTerm)
        : target{ expectedTarget }
        , regularization{ regularizationTerm }
    {}

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED T MeanSquaredError<T, NumberOfFeatures>::Cost(const Vector& predictions)
    {
        T sum{ 0 };

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
        {
            const T difference{ predictions[i] - target[i] };
            sum += difference * difference;
        }

        return sum * inverseSize + regularization.Calculate(predictions);
    }

    template<typename T, std::size_t NumberOfFeatures>
    OPTIMIZE_FOR_SPEED typename MeanSquaredError<T, NumberOfFeatures>::Vector MeanSquaredError<T, NumberOfFeatures>::Gradient(const Vector& predictions)
    {
        const Vector regularizationGradient{ regularization.Gradient(predictions) };
        Vector gradient{};

        for (std::size_t i = 0; i < NumberOfFeatures; ++i)
            gradient[i] = T{ 2 } * (predictions[i] - target[i]) * inverseSize + regularizationGradient[i];

        return gradient;
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class MeanSquaredError<float, 4>;
#endif
}
