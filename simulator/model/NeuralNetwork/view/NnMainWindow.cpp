#include "simulator/model/NeuralNetwork/view/NnMainWindow.hpp"
#include "simulator/shell/Guard.hpp"
#include "ui/theme/Theme.hpp"
#include <array>

namespace simulator::model::nn::view
{
    namespace
    {
        constexpr std::array<ui::shell::PageSpec, 2> pages{
            ui::shell::PageSpec{ "Training Loss" },
            ui::shell::PageSpec{ "Predictions" }
        };

        const ui::shell::ShellSpec shellSpec{
            "Neural Network Simulator",
            ui::Size{ 1200.0f, 700.0f },
            350.0f,
            pages,
            "Configure network parameters and press Train"
        };
    }

    NnMainWindow::NnMainWindow(QWidget* parent)
        : QMainWindow(parent)
        , formView(new ui::backend::qt::QtFormView{ this })
        , shell(*this, shellSpec)
        , lossView(new ui::backend::qt::QtPaintedWidget{ lossChart, this })
        , predictionView(new ui::backend::qt::QtPaintedWidget{ predictionChart, this })
    {
        formView->Build(form.Model());
        shell.SetPanel(formView);

        lossView->SetPanCursorEnabled(true);
        predictionView->SetPanCursorEnabled(true);

        shell.SetPage(0, lossView);
        shell.SetPage(1, predictionView);

        form.Model().onActionTriggered = [this](ui::model::ActionId)
        {
            OnComputeRequested();
        };
    }

    void NnMainWindow::OnComputeRequested()
    {
        shell::Guard(shell, "Training Error", [this]
            {
                if (form.Model().Validate())
                {
                    shell.SetStatus("A parameter is outside its allowed range");
                    return;
                }

                const auto config{ form.BuildConfiguration() };

                NnSimulator simulator;
                simulator.Configure(config);
                const auto result{ simulator.Run() };

                if (!result)
                {
                    shell.SetStatus("The network or sample count is too small to train");
                    return;
                }

                ShowResult(config, *result);
            });
    }

    void NnMainWindow::ShowResult(const NnSimulator::Configuration& config, const NnResult& result)
    {
        const auto& theme = ui::theme::Current();

        lossChart.SetAxisValues(result.epochIndex);
        lossChart.SetPanels({
            {
                "Training Loss (MSE)",
                "Loss",
                {
                    { "MSE", theme.Series(1), result.lossHistory },
                },
                1,
            },
        });

        if (config.nn.demo == DemoType::Xor)
            predictionAxis.Select(xorAxis);
        else
            predictionAxis.Select(sineAxis);

        predictionChart.SetAxisValues(result.predictionAxis);
        predictionChart.SetPanels({
            {
                "Target vs Prediction",
                "Value",
                {
                    { "Target", theme.Series(0), result.predictionTargets },
                    { "Prediction", theme.Series(1), result.predictionOutputs },
                },
                1,
            },
        });

        lossView->update();
        predictionView->update();

        const auto finalLoss{ static_cast<double>(result.finalLoss) };
        const auto status{ QString("Training complete: %1 epochs, final MSE = %2").arg(config.nn.epochs).arg(finalLoss, 0, 'g', 6) };
        shell.SetStatus(status.toStdString());
    }
}
