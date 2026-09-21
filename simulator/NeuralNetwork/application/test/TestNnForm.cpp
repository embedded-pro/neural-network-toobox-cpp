#include "simulator/NeuralNetwork/application/NnForm.hpp"
#include <gmock/gmock.h>

namespace
{
    using simulator::neural_network::nn::NnForm;
    namespace field = simulator::neural_network::nn::field;

    class NnFormTest
        : public ::testing::Test
    {
    protected:
        NnForm form;
    };
}

TEST_F(NnFormTest, TheDefaultsMatchThePanelThisReplacedRatherThanTheStruct)
{
    const auto config = form.BuildConfiguration();

    EXPECT_EQ(config.nn.demo, simulator::neural_network::nn::DemoType::Xor);
    EXPECT_EQ(config.nn.hiddenSize, 8u);
    EXPECT_NEAR(config.nn.learningRate, 0.5f, 1e-4f);
    EXPECT_EQ(config.nn.epochs, 1000u);
    EXPECT_EQ(config.nn.sineSamples, 50u);
}

TEST_F(NnFormTest, TheDemoComesFromTheOptionDataNotItsPosition)
{
    form.Model().SetSelection(field::demo, 1);

    EXPECT_EQ(form.BuildConfiguration().nn.demo, simulator::neural_network::nn::DemoType::SineApproximation);
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

TEST_F(NnFormTest, TheWholeNumberFieldsAreIntegers)
{
    for (const auto id : { field::hiddenSize, field::epochs, field::sineSamples })
        EXPECT_EQ(form.Model().Field(id).kind, ui::model::FieldKind::Integer);
}

TEST_F(NnFormTest, EachStepMatchesTheSpinBoxItReplaces)
{
    EXPECT_NEAR(form.Model().Field(field::hiddenSize).number.step, 2.0, 1e-9);
    EXPECT_NEAR(form.Model().Field(field::epochs).number.step, 100.0, 1e-9);
    EXPECT_NEAR(form.Model().Field(field::sineSamples).number.step, 10.0, 1e-9);
    EXPECT_NEAR(form.Model().Field(field::learningRate).number.step, 0.01, 1e-9);
}
