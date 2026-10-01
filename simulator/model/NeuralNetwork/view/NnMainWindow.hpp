#pragma once

#include "simulator/model/NeuralNetwork/application/NnForm.hpp"
#include "simulator/model/NeuralNetwork/view/SelectableAxis.hpp"
#include "ui/backend/qt/QtAppShell.hpp"
#include "ui/backend/qt/QtFormView.hpp"
#include "ui/backend/qt/QtPaintedWidget.hpp"
#include "ui/charts/ChartCore.hpp"
#include "ui/charts/LinearAxis.hpp"
#include <QMainWindow>

namespace simulator::model::nn::view
{
    class NnMainWindow
        : public QMainWindow
    {
        Q_OBJECT

    public:
        explicit NnMainWindow(QWidget* parent = nullptr);

    private:
        void OnComputeRequested();
        void ShowResult(const NnSimulator::Configuration& config, const NnResult& result);

        NnForm form;
        ui::backend::qt::QtFormView* formView;
        ui::backend::qt::QtAppShell shell;

        ui::charts::LinearAxis epochAxis{ "Epoch", 5, 0, "e = ", "" };
        ui::charts::LinearAxis sineAxis{ "Input x", 5, 2, "x = ", "" };
        ui::charts::LinearAxis xorAxis{ "Sample: 0 = (0,0), 1 = (0,1), 2 = (1,0), 3 = (1,1)", 3, 0, "sample ", "" };
        SelectableAxis predictionAxis{ xorAxis };
        ui::charts::ChartCore lossChart{ epochAxis, ui::charts::ChartConfig{} };
        ui::charts::ChartCore predictionChart{ predictionAxis, ui::charts::ChartConfig{} };

        ui::backend::qt::QtPaintedWidget* lossView;
        ui::backend::qt::QtPaintedWidget* predictionView;
    };
}
