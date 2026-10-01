#pragma once

#include "neural_network/losses/Loss.hpp"
#include "numerical/math/Matrix.hpp"
#include "numerical/regularization/Regularization.hpp"
#include <cstddef>
#include <gmock/gmock.h>

namespace neural_network::test_support
{
    inline constexpr float finiteDifferenceStep{ 1e-3f };

    template<std::size_t Size>
    class RegularizationMock
        : public regularization::Regularization<float, Size>
    {
    public:
        using Vector = typename regularization::Regularization<float, Size>::Vector;

        MOCK_METHOD(float, Calculate, (const Vector& parameters), (const, override));
        MOCK_METHOD(Vector, Gradient, (const Vector& parameters), (const, override));
    };

    template<std::size_t Size>
    math::Vector<float, Size> CentralDifferenceGradient(Loss<float, Size>& loss, const math::Vector<float, Size>& point)
    {
        math::Vector<float, Size> gradient{};

        for (std::size_t i = 0; i < Size; ++i)
        {
            auto plus{ point };
            auto minus{ point };
            plus[i] += finiteDifferenceStep;
            minus[i] -= finiteDifferenceStep;
            gradient[i] = (loss.Cost(plus) - loss.Cost(minus)) / (2.0f * finiteDifferenceStep);
        }

        return gradient;
    }
}
