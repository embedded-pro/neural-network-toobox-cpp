#pragma once

#include "simulator/NeuralNetwork/application/NnForm.hpp"
#include "ui/backend/qt/QtAppShell.hpp"
#include "ui/backend/qt/QtFormView.hpp"
#include "ui/backend/qt/QtPaintedWidget.hpp"
#include "ui/charts/ChartCore.hpp"
#include "ui/charts/LinearAxis.hpp"
#include <QMainWindow>

namespace simulator::neural_network::nn::view
{
    class NnMainWindow
        : public QMainWindow
    {
        Q_OBJECT

    public:
        explicit NnMainWindow(QWidget* parent = nullptr);

    private:
        void OnComputeRequested();

        NnForm form;
        ui::backend::qt::QtFormView* formView;
        ui::backend::qt::QtAppShell shell;

        ui::charts::LinearAxis epochAxis{ "Epoch", 5, 0, "e = ", "" };
        ui::charts::LinearAxis inputAxis{ "Input", 5, 2, "x = ", "" };
        ui::charts::ChartCore lossChart{ epochAxis, ui::charts::ChartConfig{} };
        ui::charts::ChartCore predictionChart{ inputAxis, ui::charts::ChartConfig{} };

        ui::backend::qt::QtPaintedWidget* lossView;
        ui::backend::qt::QtPaintedWidget* predictionView;
    };
}
