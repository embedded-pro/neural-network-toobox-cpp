#include "neural_network/activation/LeakyReLU.hpp"
#include "neural_network/activation/test/ActivationFiniteDifference.hpp"
#include "numerical/math/Tolerance.hpp"
#include <array>
#include <gtest/gtest.h>

namespace
{
    class TestLeakyReLU
        : public ::testing::Test
    {
    protected:
        neural_network::LeakyReLU<float> activation{ 0.2f };
    };
}

TEST_F(TestLeakyReLU, ForwardScalesNegativeInputBySlope)
{
    EXPECT_NEAR(activation.Forward(3.0f), 3.0f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(-2.0f), -0.4f, math::Tolerance<float>());
}

TEST_F(TestLeakyReLU, BackwardIsExactlyOneForPositiveAndSlopeOtherwise)
{
    EXPECT_FLOAT_EQ(activation.Backward(1.5f), 1.0f);
    EXPECT_FLOAT_EQ(activation.Backward(-1.5f), 0.2f);
}

TEST_F(TestLeakyReLU, DefaultSlopeIsOneHundredth)
{
    const neural_network::LeakyReLU<float> defaultActivation;

    EXPECT_NEAR(defaultActivation.Forward(-1.0f), -0.01f, math::Tolerance<float>());
}

TEST_F(TestLeakyReLU, BackwardVectorMatchesFiniteDifference)
{
    const std::array<float, 4> input{ -2.0f, -0.5f, 0.7f, 1.5f };
    const std::array<float, 4> upstream{ 0.3f, -1.2f, 0.8f, -0.4f };

    const auto analytic{ neural_network::test_support::AnalyticGradient(activation, input, upstream) };
    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(activation, input, upstream) };

    for (std::size_t i = 0; i < input.size(); ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}
