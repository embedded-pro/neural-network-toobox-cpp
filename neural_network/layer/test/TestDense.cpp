#include "neural_network/activation/LeakyReLU.hpp"
#include "neural_network/activation/Tanh.hpp"
#include "neural_network/layer/Dense.hpp"
#include "numerical/math/Tolerance.hpp"
#include <cstddef>
#include <gtest/gtest.h>

namespace
{
    constexpr float finiteDifferenceStep{ 1e-3f };

    class TestDense
        : public ::testing::Test
    {
    protected:
        using DenseLayer = neural_network::Dense<float, 3, 2>;
        using InputVector = DenseLayer::InputVector;
        using OutputVector = DenseLayer::OutputVector;
        using ParameterVector = DenseLayer::ParameterVector;

        float ProjectedOutput(DenseLayer& layer, const InputVector& input, const OutputVector& upstream)
        {
            layer.Forward(input);
            return upstream[0] * layer.Output()[0] + upstream[1] * layer.Output()[1];
        }

        const DenseLayer::WeightMatrix weights{ { 0.1f, -0.2f, 0.3f }, { 0.4f, 0.5f, -0.6f } };
        const InputVector input{ 1.0f, 2.0f, 3.0f };
        neural_network::LeakyReLU<float> leakyRelu{ 0.1f };
        neural_network::Tanh<float> tanhActivation;
    };
}

TEST_F(TestDense, ForwardAppliesActivationToAffineMap)
{
    DenseLayer layer{ weights, leakyRelu };

    layer.Forward(input);

    EXPECT_NEAR(layer.Output()[0], 0.6f, math::Tolerance<float>());
    EXPECT_NEAR(layer.Output()[1], -0.04f, math::Tolerance<float>());
}

TEST_F(TestDense, ParametersAreRowMajorWeightsFollowedByZeroBiases)
{
    const DenseLayer layer{ weights, leakyRelu };
    const ParameterVector expected{ 0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f, 0.0f, 0.0f };

    const auto& parameters{ layer.Parameters() };

    for (std::size_t i = 0; i < ParameterVector::size; ++i)
        EXPECT_NEAR(parameters[i], expected[i], math::Tolerance<float>());
}

TEST_F(TestDense, SetParametersUsesSameLayoutForForward)
{
    DenseLayer layer{ weights, leakyRelu };
    const ParameterVector newParameters{ 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.5f, -1.0f };

    layer.SetParameters(newParameters);
    layer.Forward(input);

    EXPECT_NEAR(layer.Output()[0], 1.5f, math::Tolerance<float>());
    EXPECT_NEAR(layer.Output()[1], 5.0f, math::Tolerance<float>());
    for (std::size_t i = 0; i < ParameterVector::size; ++i)
        EXPECT_NEAR(layer.Parameters()[i], newParameters[i], math::Tolerance<float>());
}

TEST_F(TestDense, BackwardInputGradientMatchesFiniteDifference)
{
    DenseLayer layer{ weights, tanhActivation };
    layer.SetParameters(ParameterVector{ 0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f, 0.2f, -0.1f });
    const InputVector point{ 0.3f, -0.7f, 1.1f };
    const OutputVector upstream{ 0.8f, -1.3f };

    InputVector numeric{};
    for (std::size_t j = 0; j < InputVector::size; ++j)
    {
        InputVector plus{ point };
        InputVector minus{ point };
        plus[j] += finiteDifferenceStep;
        minus[j] -= finiteDifferenceStep;
        numeric[j] = (ProjectedOutput(layer, plus, upstream) - ProjectedOutput(layer, minus, upstream)) / (2.0f * finiteDifferenceStep);
    }

    layer.Forward(point);
    const auto& analytic{ layer.Backward(upstream) };

    for (std::size_t j = 0; j < InputVector::size; ++j)
        EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>());
}
