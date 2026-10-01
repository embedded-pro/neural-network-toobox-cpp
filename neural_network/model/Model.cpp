#include "neural_network/model/Model.hpp"

namespace neural_network
{
    template class Model<float, 2, 1, Dense<float, 2, 3>, Dense<float, 3, 1>>;
}
