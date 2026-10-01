#include "neural_network/losses/BinaryCrossEntropy.hpp"
#include "neural_network/losses/test/LossTestSupport.hpp"
#include "numerical/math/Tolerance.hpp"
#include <gmock/gmock.h>

namespace
{
    class TestBinaryCrossEntropy
        : public ::testing::Test
    {
    protected:
        static constexpr std::size_t size{ 4 };
        using Vector = neural_network::BinaryCrossEntropy<float, size>::Vector;

        ::testing::StrictMock<neural_network::test_support::RegularizationMock<size>> regularization;
        const Vector target{ 1.0f, 0.0f, 1.0f, 0.0f };
        const Vector predictions{ 0.9f, 0.2f, 0.6f, 0.3f };
        neural_network::BinaryCrossEntropy<float, size> loss{ target, regularization };
    };
}

TEST_F(TestBinaryCrossEntropy, CostIsMeanCrossEntropyPlusRegularization)
{
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillOnce(::testing::Return(0.1f));

    EXPECT_NEAR(loss.Cost(predictions), 0.3990012f, math::Tolerance<float>());
}

TEST_F(TestBinaryCrossEntropy, GradientMatchesReferencePlusRegularizationGradient)
{
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{ 0.01f, 0.02f, 0.03f, 0.04f }));

    const auto gradient{ loss.Gradient(predictions) };

    EXPECT_NEAR(gradient[0], -0.2677778f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[1], 0.3325f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[2], -0.3866667f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[3], 0.3971429f, math::Tolerance<float>());
}

TEST_F(TestBinaryCrossEntropy, GradientMatchesFiniteDifferenceOfCost)
{
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillRepeatedly(::testing::Return(0.0f));
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{}));

    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(loss, predictions) };
    const auto analytic{ loss.Gradient(predictions) };

    for (std::size_t i = 0; i < size; ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}

TEST_F(TestBinaryCrossEntropy, SaturatedProbabilitiesGiveFiniteCostAndGradient)
{
    const Vector saturated{ 1.0f, 0.0f, 0.0f, 1.0f };
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillOnce(::testing::Return(0.0f));
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{}));

    EXPECT_NEAR(loss.Cost(saturated), 8.01512f, 10.0f * math::Tolerance<float>());

    const auto gradient{ loss.Gradient(saturated) };
    EXPECT_NEAR(gradient[0], -0.25f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[1], 0.25f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[2], -2.5e6f, 2.5e6f * math::Tolerance<float>());
    EXPECT_NEAR(gradient[3], 2097152.0f, 2097152.0f * math::Tolerance<float>());
}
