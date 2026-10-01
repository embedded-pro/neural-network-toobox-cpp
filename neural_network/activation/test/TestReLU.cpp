#include "neural_network/activation/ReLU.hpp"
#include "neural_network/activation/test/ActivationFiniteDifference.hpp"
#include "numerical/math/Tolerance.hpp"
#include <array>
#include <gtest/gtest.h>

namespace
{
    class TestReLU
        : public ::testing::Test
    {
    protected:
        neural_network::ReLU<float> activation;
    };
}

TEST_F(TestReLU, ForwardPassesPositiveAndZeroesNonPositive)
{
    EXPECT_NEAR(activation.Forward(2.5f), 2.5f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(-1.5f), 0.0f, math::Tolerance<float>());
    EXPECT_NEAR(activation.Forward(0.0f), 0.0f, math::Tolerance<float>());
}

TEST_F(TestReLU, BackwardIsExactlyOneForPositiveAndZeroOtherwise)
{
    EXPECT_FLOAT_EQ(activation.Backward(3.0f), 1.0f);
    EXPECT_FLOAT_EQ(activation.Backward(-3.0f), 0.0f);
    EXPECT_FLOAT_EQ(activation.Backward(0.0f), 0.0f);
}

TEST_F(TestReLU, BackwardVectorMatchesFiniteDifference)
{
    const std::array<float, 4> input{ -2.0f, -0.5f, 0.7f, 1.5f };
    const std::array<float, 4> upstream{ 0.3f, -1.2f, 0.8f, -0.4f };

    const auto analytic{ neural_network::test_support::AnalyticGradient(activation, input, upstream) };
    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(activation, input, upstream) };

    for (std::size_t i = 0; i < input.size(); ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}
