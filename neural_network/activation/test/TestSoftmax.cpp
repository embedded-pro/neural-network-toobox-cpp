#include "neural_network/activation/Softmax.hpp"
#include "neural_network/activation/test/ActivationFiniteDifference.hpp"
#include "numerical/math/Tolerance.hpp"
#include <array>
#include <gtest/gtest.h>

namespace
{
    class TestSoftmax
        : public ::testing::Test
    {
    protected:
        neural_network::Softmax<float> activation;
    };
}

TEST_F(TestSoftmax, ForwardVectorMatchesReferenceDistribution)
{
    const std::array<float, 3> input{ 1.0f, 2.0f, 3.0f };
    std::array<float, 3> output{};

    activation.ForwardVector(output, input);

    EXPECT_NEAR(output[0], 0.0900306f, math::Tolerance<float>());
    EXPECT_NEAR(output[1], 0.2447285f, math::Tolerance<float>());
    EXPECT_NEAR(output[2], 0.6652410f, math::Tolerance<float>());
}

TEST_F(TestSoftmax, ForwardVectorIsStableForLargeLogitsAndSumsToOne)
{
    const std::array<float, 3> input{ 1000.0f, 0.0f, -1000.0f };
    std::array<float, 3> output{};

    activation.ForwardVector(output, input);

    EXPECT_NEAR(output[0], 1.0f, math::Tolerance<float>());
    EXPECT_NEAR(output[1], 0.0f, math::Tolerance<float>());
    EXPECT_NEAR(output[2], 0.0f, math::Tolerance<float>());
    EXPECT_NEAR(output[0] + output[1] + output[2], 1.0f, math::Tolerance<float>());
}

TEST_F(TestSoftmax, BackwardVectorMatchesFiniteDifference)
{
    const std::array<float, 3> input{ 0.5f, -1.0f, 2.0f };
    const std::array<float, 3> upstream{ 1.0f, -2.0f, 0.5f };

    const auto analytic{ neural_network::test_support::AnalyticGradient(activation, input, upstream) };
    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(activation, input, upstream) };

    for (std::size_t i = 0; i < input.size(); ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}

TEST_F(TestSoftmax, ScalarForwardIsSingleElementSoftmax)
{
    EXPECT_FLOAT_EQ(activation.Forward(-3.0f), 1.0f);
    EXPECT_FLOAT_EQ(activation.Backward(-3.0f), 0.0f);
}
