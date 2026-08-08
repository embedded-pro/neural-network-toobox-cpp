#include "simulator/NeuralNetwork/view/NnMainWindow.hpp"
#include <QApplication>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    simulator::neural_network::nn::view::NnMainWindow window;
    window.show();

    return app.exec();
}
