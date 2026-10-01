#include "neural_network/losses/MeanAbsoluteError.hpp"
#include "neural_network/losses/test/LossTestSupport.hpp"
#include "numerical/math/Tolerance.hpp"
#include <gmock/gmock.h>

namespace
{
    class TestMeanAbsoluteError
        : public ::testing::Test
    {
    protected:
        static constexpr std::size_t size{ 4 };
        using Vector = neural_network::MeanAbsoluteError<float, size>::Vector;

        ::testing::StrictMock<neural_network::test_support::RegularizationMock<size>> regularization;
        const Vector target{ 0.5f, -1.0f, 2.0f, 0.0f };
        const Vector predictions{ 1.0f, -1.5f, 2.5f, 1.0f };
        neural_network::MeanAbsoluteError<float, size> loss{ target, regularization };
    };
}

TEST_F(TestMeanAbsoluteError, CostIsMeanAbsoluteErrorPlusRegularization)
{
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillOnce(::testing::Return(0.1f));

    EXPECT_NEAR(loss.Cost(predictions), 0.725f, math::Tolerance<float>());
}

TEST_F(TestMeanAbsoluteError, GradientIsSignOverSizePlusRegularizationGradient)
{
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{ 0.01f, 0.02f, 0.03f, 0.04f }));

    const auto gradient{ loss.Gradient(predictions) };

    EXPECT_NEAR(gradient[0], 0.26f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[1], -0.23f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[2], 0.28f, math::Tolerance<float>());
    EXPECT_NEAR(gradient[3], 0.29f, math::Tolerance<float>());
}

TEST_F(TestMeanAbsoluteError, GradientMatchesFiniteDifferenceOfCost)
{
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillRepeatedly(::testing::Return(0.0f));
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{}));

    const auto numeric{ neural_network::test_support::CentralDifferenceGradient(loss, predictions) };
    const auto analytic{ loss.Gradient(predictions) };

    for (std::size_t i = 0; i < size; ++i)
        EXPECT_NEAR(analytic[i], numeric[i], math::Tolerance<float>());
}

TEST_F(TestMeanAbsoluteError, ZeroErrorGivesZeroCostAndZeroGradient)
{
    EXPECT_CALL(regularization, Calculate(::testing::_)).WillOnce(::testing::Return(0.0f));
    EXPECT_CALL(regularization, Gradient(::testing::_)).WillOnce(::testing::Return(Vector{}));

    EXPECT_FLOAT_EQ(loss.Cost(target), 0.0f);
    const auto gradient{ loss.Gradient(target) };
    for (std::size_t i = 0; i < size; ++i)
        EXPECT_FLOAT_EQ(gradient[i], 0.0f);
}
