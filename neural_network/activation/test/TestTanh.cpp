#include "neural_network/activation/Tanh.hpp"
#include "neural_network/activation/test/ActivationFiniteDifference.hpp"
#include "numerical/math/Tolerance.hpp"
#include <array>
#include <gtest/gtest.h>

namespace
{
    class TestTanh
        : public ::testing::Test
    {
    protected:
        neural_network::Tanh<float> activation;
    };
}

TEST_F(TestTanh, ForwardMatchesReferenceValues)
{
    EXPECT_NEAR(activation.Forward(0.5f), 0.4621172f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(-1.0f), -0.7615942f, math::Tolerance<float>());
}

TEST_F(TestTanh, BackwardAtZeroIsExactlyOne)
{
    EXPECT_FLOAT_EQ(activation.Backward(0.0f), 1.0f);
}

TEST_F(TestTanh, BackwardMatchesFiniteDifference)
{
    for (const float x : { -2.0f, -0.3f, 0.7f, 1.5f })
        EXPECT_NEAR(activation.Backward(x), neural_network::test_support::CentralDifference(activation, x), math::Tolerance<float>());
}

TEST_F(TestTanh, BackwardVectorMatchesFiniteDifference)
{
    const std::array<float, 4> input{ -2.0f, -0.3f, 0.7f, 1.5f };
    const std::array<float, 4> upstream{ 0.3f, -1.2f, 0.8f, -0.4f };

    const auto analytic{ neural_network::test_support::AnalyticGradient(activation, input, upstream) };
    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(activation, input, upstream) };

    for (std::size_t i = 0; i < input.size(); ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}

TEST_F(TestTanh, SaturatesWithoutNegativeDerivative)
{
    EXPECT_NEAR(activation.Forward(20.0f), 1.0f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(-20.0f), -1.0f, math::Tolerance<float>());
    EXPECT_GE(activation.Backward(20.0f), 0.0f);
    EXPECT_GE(activation.Backward(-20.0f), 0.0f);
    EXPECT_NEAR(activation.Backward(20.0f), 0.0f, math::Tolerance<float>());
}
