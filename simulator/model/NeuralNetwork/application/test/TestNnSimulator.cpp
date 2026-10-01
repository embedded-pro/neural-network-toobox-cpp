#include "simulator/model/NeuralNetwork/application/NnSimulator.hpp"
#include <cmath>
#include <gmock/gmock.h>

namespace
{
    using simulator::model::nn::DemoType;
    using simulator::model::nn::NnResult;
    using simulator::model::nn::NnSimulator;

    class NnSimulatorTest
        : public ::testing::Test
    {
    protected:
        std::optional<NnResult> Run()
        {
            simulator.Configure(configuration);
            return simulator.Run();
        }

        NnSimulator::Configuration configuration{};
        NnSimulator simulator;
    };
}

TEST_F(NnSimulatorTest, TheXorDemoConvergesWithTheDefaultsAndFixedSeed)
{
    const auto result = Run();

    ASSERT_TRUE(result.has_value());
    EXPECT_LT(result->finalLoss, 1e-3f);

    ASSERT_EQ(result->predictionOutputs.size(), 4u);
    for (std::size_t s = 0; s < 4; ++s)
        EXPECT_EQ(std::round(result->predictionOutputs[s]), result->predictionTargets[s]) << "sample " << s;
}

TEST_F(NnSimulatorTest, TheSineDemoFitsWithTheDefaults)
{
    configuration.nn.demo = DemoType::SineApproximation;

    const auto result = Run();

    ASSERT_TRUE(result.has_value());
    EXPECT_LT(result->finalLoss, 1e-3f);

    for (std::size_t s = 0; s < result->predictionOutputs.size(); ++s)
        EXPECT_NEAR(result->predictionOutputs[s], result->predictionTargets[s], 0.05f) << "sample " << s;
}

TEST_F(NnSimulatorTest, TheLossHistoryHasOneEntryPerEpochAndThePredictionsOnePerSample)
{
    configuration.nn.demo = DemoType::SineApproximation;
    configuration.nn.epochs = 37;
    configuration.nn.sineSamples = 23;

    const auto result = Run();

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->epochIndex.size(), 37u);
    EXPECT_EQ(result->lossHistory.size(), 37u);
    EXPECT_EQ(result->predictionAxis.size(), 23u);
    EXPECT_EQ(result->predictionTargets.size(), 23u);
    EXPECT_EQ(result->predictionOutputs.size(), 23u);
}

TEST_F(NnSimulatorTest, TheFinalLossIsTheMeanSquaredErrorOfTheReturnedPredictions)
{
    configuration.nn.epochs = 10;

    const auto result = Run();

    ASSERT_TRUE(result.has_value());

    auto squaredError{ 0.0f };
    for (std::size_t s = 0; s < result->predictionOutputs.size(); ++s)
    {
        const auto error{ result->predictionOutputs[s] - result->predictionTargets[s] };
        squaredError += error * error;
    }

    EXPECT_NEAR(result->finalLoss, squaredError / 4.0f, 1e-6f);
    EXPECT_LT(result->finalLoss, result->lossHistory.back());
}

TEST_F(NnSimulatorTest, XorPredictionsArePlottedAgainstTheirSampleIndex)
{
    configuration.nn.epochs = 10;

    const auto result = Run();

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(result->predictionAxis, ::testing::ElementsAre(0.0f, 1.0f, 2.0f, 3.0f));
}

TEST_F(NnSimulatorTest, SinePredictionsArePlottedAgainstTheirInputFromZeroToOne)
{
    configuration.nn.demo = DemoType::SineApproximation;
    configuration.nn.epochs = 10;
    configuration.nn.sineSamples = 5;

    const auto result = Run();

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(result->predictionAxis, ::testing::ElementsAre(0.0f, 0.25f, 0.5f, 0.75f, 1.0f));
}

TEST_F(NnSimulatorTest, ASineRunNeedsAtLeastTwoSamples)
{
    configuration.nn.demo = DemoType::SineApproximation;

    configuration.nn.sineSamples = 0;
    EXPECT_FALSE(Run().has_value());

    configuration.nn.sineSamples = 1;
    EXPECT_FALSE(Run().has_value());

    configuration.nn.sineSamples = 2;
    EXPECT_TRUE(Run().has_value());
}

TEST_F(NnSimulatorTest, AnEmptyHiddenLayerIsRejected)
{
    configuration.nn.hiddenSize = 0;

    EXPECT_FALSE(Run().has_value());
}
