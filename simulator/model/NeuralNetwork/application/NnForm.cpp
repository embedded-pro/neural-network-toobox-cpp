#include "simulator/model/NeuralNetwork/application/NnForm.hpp"

namespace simulator::model::nn
{
    namespace
    {
        using ui::model::ActionSpec;
        using ui::model::Condition;
        using ui::model::FieldKind;
        using ui::model::FieldSpec;
        using ui::model::GroupSpec;
        using ui::model::OptionSpec;

        constexpr NnConfig defaults{};

        constexpr std::array<OptionSpec, 2> demoTypes{
            OptionSpec{ "XOR Classification", static_cast<std::int64_t>(DemoType::Xor) },
            OptionSpec{ "Sine Approximation", static_cast<std::int64_t>(DemoType::SineApproximation) }
        };

        constexpr Condition sineSelected{ field::demo, 0b10 };

        constexpr std::array<GroupSpec, 2> groups{
            GroupSpec{ field::selection, "Demo Selection", {} },
            GroupSpec{ field::network, "Network Parameters", {} }
        };

        constexpr std::array<FieldSpec, 5> fields{
            FieldSpec{ field::demo, field::selection, FieldKind::Choice, "Problem:", "", {}, demoTypes, {}, {} },
            FieldSpec{ field::hiddenSize, field::network, FieldKind::Integer, "Hidden Layer Size:", "", { 2.0, 64.0, 2.0, static_cast<double>(defaults.hiddenSize), 0 }, {}, {}, {} },
            FieldSpec{ field::learningRate, field::network, FieldKind::Number, "Learning Rate:", "", { 0.001, 10.0, 0.01, static_cast<double>(defaults.learningRate), 3 }, {}, {}, {} },
            FieldSpec{ field::epochs, field::network, FieldKind::Integer, "Epochs:", "", { 10.0, 10000.0, 100.0, static_cast<double>(defaults.epochs), 0 }, {}, {}, {} },
            FieldSpec{ field::sineSamples, field::network, FieldKind::Integer, "Sine Samples:", "", { 10.0, 500.0, 10.0, static_cast<double>(defaults.sineSamples), 0 }, {}, {}, sineSelected }
        };

        constexpr std::array<ActionSpec, 1> actions{
            ActionSpec{ field::train, "Train", ui::theme::ButtonRole::Primary, 0 }
        };
    }

    NnForm::NnForm()
        : spec{ groups, fields, actions, {} }
        , model{ spec, values, {} }
    {}

    ui::model::FormModel& NnForm::Model()
    {
        return model;
    }

    const ui::model::FormModel& NnForm::Model() const
    {
        return model;
    }

    NnSimulator::Configuration NnForm::BuildConfiguration() const
    {
        NnSimulator::Configuration config{};

        config.nn.demo = static_cast<DemoType>(model.SelectedData(field::demo));
        config.nn.hiddenSize = model.Count(field::hiddenSize);
        config.nn.learningRate = model.Float(field::learningRate);
        config.nn.epochs = model.Count(field::epochs);
        config.nn.sineSamples = model.Count(field::sineSamples);

        return config;
    }
}
