#include "neural_network/activation/Sigmoid.hpp"
#include "neural_network/activation/test/ActivationFiniteDifference.hpp"
#include "numerical/math/Tolerance.hpp"
#include <array>
#include <gtest/gtest.h>

namespace
{
    class TestSigmoid
        : public ::testing::Test
    {
    protected:
        neural_network::Sigmoid<float> activation;
    };
}

TEST_F(TestSigmoid, ForwardMatchesReferenceValues)
{
    EXPECT_NEAR(activation.Forward(0.0f), 0.5f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(2.0f), 0.8807971f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(-2.0f), 0.1192029f, math::Tolerance<float>());
}

TEST_F(TestSigmoid, BackwardAtZeroIsExactlyOneQuarter)
{
    EXPECT_FLOAT_EQ(activation.Backward(0.0f), 0.25f);
}

TEST_F(TestSigmoid, BackwardMatchesFiniteDifference)
{
    for (const float x : { -3.0f, -0.5f, 1.0f, 4.0f })
        EXPECT_NEAR(activation.Backward(x), neural_network::test_support::CentralDifference(activation, x), math::Tolerance<float>());
}

TEST_F(TestSigmoid, BackwardVectorMatchesFiniteDifference)
{
    const std::array<float, 4> input{ -3.0f, -0.5f, 1.0f, 4.0f };
    const std::array<float, 4> upstream{ 0.3f, -1.2f, 0.8f, -0.4f };

    const auto analytic{ neural_network::test_support::AnalyticGradient(activation, input, upstream) };
    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(activation, input, upstream) };

    for (std::size_t i = 0; i < input.size(); ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}

TEST_F(TestSigmoid, SaturatesWithoutOverflowOrNegativeDerivative)
{
    EXPECT_NEAR(activation.Forward(50.0f), 1.0f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(-50.0f), 0.0f, math::Tolerance<float>());
    EXPECT_GE(activation.Backward(50.0f), 0.0f);
    EXPECT_GE(activation.Backward(-50.0f), 0.0f);
    EXPECT_NEAR(activation.Backward(50.0f), 0.0f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Backward(-50.0f), 0.0f, math::Tolerance<float>());
}
