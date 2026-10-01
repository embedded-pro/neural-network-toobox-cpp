#include "neural_network/activation/LeakyReLU.hpp"
#include "neural_network/activation/Tanh.hpp"
#include "neural_network/layer/Dense.hpp"
#include "neural_network/losses/Loss.hpp"
#include "neural_network/model/Model.hpp"
#include "numerical/math/Tolerance.hpp"
#include "numerical/optimization/Optimizer.hpp"
#include <cstddef>
#include <gmock/gmock.h>

namespace
{
    constexpr float finiteDifferenceStep{ 1e-3f };

    template<std::size_t Size>
    class OptimizerMock
        : public optimization::Optimizer<float, Size>
    {
    public:
        using Base = optimization::Optimizer<float, Size>;

        MOCK_METHOD(const typename Base::Result&, Minimize, (const typename Base::Vector& initialGuess, (optimization::ObjectiveFunction<float, Size> & objective)), (override));
    };

    template<std::size_t Size>
    class LossMock
        : public neural_network::Loss<float, Size>
    {
    public:
        using Vector = typename neural_network::Loss<float, Size>::Vector;

        MOCK_METHOD(float, Cost, (const Vector& parameters), (override));
        MOCK_METHOD(Vector, Gradient, (const Vector& parameters), (override));
    };

    class TestModel
        : public ::testing::Test
    {
    protected:
        using HiddenLayer = neural_network::Dense<float, 2, 3>;
        using OutputLayer = neural_network::Dense<float, 3, 1>;
        using ModelType = neural_network::Model<float, 2, 1, HiddenLayer, OutputLayer>;
        using InputVector = ModelType::InputVector;
        using OutputVector = ModelType::OutputVector;
        using ParameterVector = ModelType::ParameterVector;

        float ModelOutput(const InputVector& input)
        {
            return model.Forward(input)[0];
        }

        neural_network::LeakyReLU<float> leakyRelu{ 0.1f };
        neural_network::Tanh<float> tanhActivation;
        const HiddenLayer::WeightMatrix hiddenWeights{ { 0.5f, -1.0f }, { 1.5f, 0.25f }, { -0.5f, 0.75f } };
        const OutputLayer::WeightMatrix outputWeights{ { 1.0f, -0.5f, 2.0f } };
        ModelType model{ neural_network::make_layer<HiddenLayer>(hiddenWeights, leakyRelu), neural_network::make_layer<OutputLayer>(outputWeights, tanhActivation) };
        const InputVector input{ 2.0f, 0.5f };
    };
}

TEST_F(TestModel, ForwardComposesDenseLayers)
{
    EXPECT_NEAR(model.Forward(input)[0], -0.8298019f, math::Tolerance<float>());
}

TEST_F(TestModel, BackwardInputGradientMatchesFiniteDifference)
{
    InputVector numeric{};
    for (std::size_t j = 0; j < InputVector::size; ++j)
    {
        InputVector plus{ input };
        InputVector minus{ input };
        plus[j] += finiteDifferenceStep;
        minus[j] -= finiteDifferenceStep;
        numeric[j] = (ModelOutput(plus) - ModelOutput(minus)) / (2.0f * finiteDifferenceStep);
    }

    static_cast<void>(model.Forward(input));
    const auto analytic{ model.Backward(OutputVector{ 1.0f }) };

    for (std::size_t j = 0; j < InputVector::size; ++j)
        EXPECT_NEAR(analytic[j], numeric[j], math::Tolerance<float>());
}

TEST_F(TestModel, GetParametersConcatenatesLayersInOrder)
{
    const ParameterVector expected{ 0.5f, -1.0f, 1.5f, 0.25f, -0.5f, 0.75f, 0.0f, 0.0f, 0.0f, 1.0f, -0.5f, 2.0f, 0.0f };

    const auto parameters{ model.GetParameters() };

    for (std::size_t i = 0; i < ModelType::TotalParameters; ++i)
        EXPECT_NEAR(parameters[i], expected[i], math::Tolerance<float>());
}

TEST_F(TestModel, SetParametersRoundTripsThroughLayers)
{
    ParameterVector parameters{};
    for (std::size_t i = 0; i < ModelType::TotalParameters; ++i)
        parameters[i] = 0.1f * static_cast<float>(i) - 0.6f;

    model.SetParameters(parameters);
    const auto retrieved{ model.GetParameters() };

    for (std::size_t i = 0; i < ModelType::TotalParameters; ++i)
        EXPECT_NEAR(retrieved[i], parameters[i], math::Tolerance<float>());
}

TEST_F(TestModel, TrainAppliesOptimizerResult)
{
    ::testing::StrictMock<OptimizerMock<ModelType::TotalParameters>> optimizer;
    ::testing::StrictMock<LossMock<ModelType::TotalParameters>> loss;
    ParameterVector optimized{};
    for (std::size_t i = 0; i < ModelType::TotalParameters; ++i)
        optimized[i] = 0.05f * static_cast<float>(i + 1);
    const OptimizerMock<ModelType::TotalParameters>::Result result{ optimized, 0.4f, 3 };
    EXPECT_CALL(optimizer, Minimize(::testing::_, ::testing::Ref(loss))).WillOnce(::testing::ReturnRef(result));

    model.Train(optimizer, loss, model.GetParameters());

    const auto parameters{ model.GetParameters() };
    for (std::size_t i = 0; i < ModelType::TotalParameters; ++i)
        EXPECT_NEAR(parameters[i], optimized[i], math::Tolerance<float>());
}
