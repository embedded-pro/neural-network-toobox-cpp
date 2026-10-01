#include "simulator/model/NeuralNetwork/application/NnForm.hpp"
#include <gmock/gmock.h>

namespace
{
    using simulator::model::nn::DemoType;
    using simulator::model::nn::NnConfig;
    using simulator::model::nn::NnForm;
    using simulator::model::nn::NnSimulator;
    namespace field = simulator::model::nn::field;

    class NnFormTest
        : public ::testing::Test
    {
    protected:
        void SetToMinimum(ui::model::FieldId id)
        {
            form.Model().SetNumber(id, form.Model().Field(id).number.minimum);
        }

        NnForm form;
    };
}

TEST_F(NnFormTest, TheDefaultsAreTheSimulatorDefaults)
{
    const auto config = form.BuildConfiguration();
    const NnConfig defaults{};

    EXPECT_EQ(config.nn.demo, defaults.demo);
    EXPECT_EQ(config.nn.hiddenSize, defaults.hiddenSize);
    EXPECT_NEAR(config.nn.learningRate, defaults.learningRate, 1e-4f);
    EXPECT_EQ(config.nn.epochs, defaults.epochs);
    EXPECT_EQ(config.nn.sineSamples, defaults.sineSamples);
}

TEST_F(NnFormTest, TheDemoComesFromTheOptionDataNotItsPosition)
{
    form.Model().SetSelection(field::demo, 0);
    EXPECT_EQ(form.BuildConfiguration().nn.demo, DemoType::Xor);

    form.Model().SetSelection(field::demo, 1);
    EXPECT_EQ(form.BuildConfiguration().nn.demo, DemoType::SineApproximation);
}

TEST_F(NnFormTest, TheSineSampleCountIsDisabledWhileTheXorDemoIsSelected)
{
    EXPECT_FALSE(form.Model().IsEnabled(field::sineSamples));

    form.Model().SetSelection(field::demo, 1);
    EXPECT_TRUE(form.Model().IsEnabled(field::sineSamples));

    form.Model().SetSelection(field::demo, 0);
    EXPECT_FALSE(form.Model().IsEnabled(field::sineSamples));
}

TEST_F(NnFormTest, TheSineSampleCountIsStillReadWhileDisabled)
{
    form.Model().SetNumber(field::sineSamples, 250.0);

    EXPECT_EQ(form.BuildConfiguration().nn.sineSamples, 250u);
}

TEST_F(NnFormTest, TheSmallestValuesTheFormAcceptsAreOnesTheSimulatorCanTrain)
{
    form.Model().SetSelection(field::demo, 1);
    SetToMinimum(field::hiddenSize);
    SetToMinimum(field::epochs);
    SetToMinimum(field::sineSamples);

    ASSERT_FALSE(form.Model().Validate().has_value());

    NnSimulator simulator;
    simulator.Configure(form.BuildConfiguration());

    EXPECT_TRUE(simulator.Run().has_value());
}
