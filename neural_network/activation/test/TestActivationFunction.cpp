#include "neural_network/activation/ActivationFunction.hpp"
#include "numerical/math/Tolerance.hpp"
#include <array>
#include <gmock/gmock.h>

namespace
{
    class ActivationFunctionMock
        : public neural_network::ActivationFunction<float>
    {
    public:
        MOCK_METHOD(float, Forward, (float x), (const, override));
        MOCK_METHOD(float, Backward, (float x), (const, override));
    };

    class TestActivationFunction
        : public ::testing::Test
    {
    protected:
        ::testing::StrictMock<ActivationFunctionMock> activation;
        const std::array<float, 2> input{ 1.0f, 2.0f };
    };
}

TEST_F(TestActivationFunction, DefaultForwardVectorAppliesForwardElementWise)
{
    EXPECT_CALL(activation, Forward(1.0f)).WillOnce(::testing::Return(10.0f));
    EXPECT_CALL(activation, Forward(2.0f)).WillOnce(::testing::Return(20.0f));

    std::array<float, 2> output{};
    activation.neural_network::ActivationFunction<float>::ForwardVector(output, input);

    EXPECT_NEAR(output[0], 10.0f, math::Tolerance<float>());
    EXPECT_NEAR(output[1], 20.0f, math::Tolerance<float>());
}

TEST_F(TestActivationFunction, DefaultBackwardVectorScalesUpstreamByPreActivationDerivative)
{
    EXPECT_CALL(activation, Backward(1.0f)).WillOnce(::testing::Return(3.0f));
    EXPECT_CALL(activation, Backward(2.0f)).WillOnce(::testing::Return(4.0f));

    const std::array<float, 2> output{};
    const std::array<float, 2> upstream{ 0.5f, -1.0f };
    std::array<float, 2> result{};
    activation.neural_network::ActivationFunction<float>::BackwardVector(result, input, output, upstream);

    EXPECT_NEAR(result[0], 1.5f, math::Tolerance<float>());
    EXPECT_NEAR(result[1], -4.0f, math::Tolerance<float>());
}
