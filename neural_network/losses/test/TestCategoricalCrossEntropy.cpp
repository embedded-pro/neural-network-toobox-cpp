#include "neural_network/losses/CategoricalCrossEntropy.hpp"
#include "neural_network/losses/test/LossTestSupport.hpp"
#include "numerical/math/Tolerance.hpp"
#include <gmock/gmock.h>

namespace
{
    class TestCategoricalCrossEntropy
        : public ::testing::Test
    {
    protected:
        static constexpr std::size_t size{ 3 };
        using Vector = neural_network::CategoricalCrossEntropy<float, size>::Vector;

        ::testing::StrictMock<neural_network::test_support::RegularizationMock<size>> regularization;
        const Vector oneHotTarget{ 0.0f, 1.0f, 0.0f };
        const Vector logits{ 1.0f, 2.0f, 3.0f };
        neural_network::CategoricalCrossEntropy<float, size> loss{ oneHotTarget, regularization };
    };
}

TEST_F(TestCategoricalCrossEntropy, CostIsNegativeLogSoftmaxOfTargetPlusRegularization)
{
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillOnce(::testing::Return(0.1f));

    EXPECT_NEAR(loss.Cost(logits), 1.5076060f, math::Tolerance<float>());
}

TEST_F(TestCategoricalCrossEntropy, GradientIsSoftmaxMinusTargetPlusRegularizationGradient)
{
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{ 0.01f, 0.02f, 0.03f }));

    const auto gradient{ loss.Gradient(logits) };

    EXPECT_NEAR(gradient[0], 0.1000306f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[1], -0.7352715f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[2], 0.6952410f, math::Tolerance<float>());
}

TEST_F(TestCategoricalCrossEntropy, GradientMatchesFiniteDifferenceForUnnormalisedTarget)
{
    neural_network::CategoricalCrossEntropy<float, size> softTargetLoss{ Vector{ 0.2f, 0.5f, 0.1f }, regularization };
    const Vector point{ 0.4f, -1.2f, 2.5f };
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillRepeatedly(::testing::Return(0.0f));
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{}));

    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(softTargetLoss, point) };
    const auto analytic{ softTargetLoss.Gradient(point) };

    for (std::size_t i = 0; i < size; ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}

TEST_F(TestCategoricalCrossEntropy, LargeLogitsGiveFiniteCostAndGradient)
{
    const Vector largeLogits{ 1000.0f, 0.0f, -1000.0f };
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillOnce(::testing::Return(0.0f));
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{}));

    EXPECT_NEAR(loss.Cost(largeLogits), 1000.0f, math::Tolerance<float>());

    const auto gradient{ loss.Gradient(largeLogits) };
    EXPECT_NEAR(gradient[0], 1.0f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[1], -1.0f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[2], 0.0f, math::Tolerance<float>());
}
