#include "simulator/NeuralNetwork/view/NnMainWindow.hpp"
#include "ui/theme/Theme.hpp"
#include <array>

namespace simulator::neural_network::nn::view
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
        auto config = form.BuildConfiguration();

        NnSimulator simulator;
        simulator.Configure(config);
        auto result = simulator.Run();

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

        predictionChart.SetAxisValues(result.predictionInputs);
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

        const auto finalLoss = result.lossHistory.empty() ? 0.0f : result.lossHistory.back();

        shell.SetStatus(QString("Training complete: %1 epochs, final loss = %2")
                .arg(config.nn.epochs)
                .arg(static_cast<double>(finalLoss), 0, 'g', 6)
                .toStdString());
    }
}
