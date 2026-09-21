#include "simulator/NeuralNetwork/view/NnMainWindow.hpp"
#include "simulator/shell/AppRunner.hpp"

int main(int argc, char* argv[])
{
    return simulator::shell::Run<simulator::neural_network::nn::view::NnMainWindow>(argc, argv, ui::theme::Light());
}
