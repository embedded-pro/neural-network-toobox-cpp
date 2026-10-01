#include "simulator/model/NeuralNetwork/application/Mlp.hpp"
#include <array>
#include <gmock/gmock.h>
#include <vector>

namespace
{
    using simulator::model::nn::Mlp;

    class MlpTest
        : public ::testing::Test
    {
    protected:
        float SumOfSquaredErrors()
        {
            auto sum{ 0.0f };

            for (std::size_t s = 0; s < inputs.size(); ++s)
            {
                const auto error{ network.Predict(inputs[s]) - targets[s] };
                sum += error * error;
            }

            return sum;
        }

        void AccumulateAll()
        {
            for (std::size_t s = 0; s < inputs.size(); ++s)
                network.Accumulate(inputs[s], targets[s]);
        }

        Mlp network{ 2, 3, 7 };
        std::array<std::array<float, 2>, 3> inputs{ { { 0.3f, -0.7f }, { -1.2f, 0.4f }, { 0.9f, 0.8f } } };
        std::array<float, 3> targets{ 0.2f, 0.9f, 0.5f };
    };
}

TEST_F(MlpTest, TheGradientMatchesCentralFiniteDifferencesOfTheSquaredError)
{
    AccumulateAll();
    const std::vector<float> analytic{ network.Gradients().begin(), network.Gradients().end() };

    constexpr auto step{ 1e-2f };
    auto parameters{ network.Parameters() };

    ASSERT_EQ(parameters.size(), 13u);

    for (std::size_t index = 0; index < parameters.size(); ++index)
    {
        const auto original{ parameters[index] };

        parameters[index] = original + step;
        const auto above{ SumOfSquaredErrors() };
        parameters[index] = original - step;
        const auto below{ SumOfSquaredErrors() };
        parameters[index] = original;

        EXPECT_NEAR(analytic[index], (above - below) / (2.0f * step), 1e-3f) << "parameter " << index;
    }
}

TEST_F(MlpTest, AStepDescendsTheMeanGradientAndClearsIt)
{
    AccumulateAll();

    const std::vector<float> before{ network.Parameters().begin(), network.Parameters().end() };
    const std::vector<float> gradient{ network.Gradients().begin(), network.Gradients().end() };

    constexpr auto learningRate{ 0.3f };
    network.Step(learningRate, inputs.size());

    for (std::size_t index = 0; index < before.size(); ++index)
    {
        EXPECT_NEAR(network.Parameters()[index], before[index] - learningRate / 3.0f * gradient[index], 1e-6f);
        EXPECT_EQ(network.Gradients()[index], 0.0f);
    }
}
