#pragma once

#include "numerical/math/Matrix.hpp"
#include "numerical/optimization/ObjectiveFunction.hpp"
#include <cstddef>
#include <type_traits>

namespace neural_network
{
    template<typename T, std::size_t NumberOfFeatures>
    class Loss
        : public optimization::ObjectiveFunction<T, NumberOfFeatures>
    {
        static_assert(std::is_floating_point_v<T>, "Loss requires a floating-point type");
        static_assert(NumberOfFeatures > 0, "Loss requires at least one feature");

    public:
        using Vector = math::Vector<T, NumberOfFeatures>;
    };
}
