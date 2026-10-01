#pragma once

#include "numerical/math/Matrix.hpp"
#include <cstddef>
#include <type_traits>

namespace neural_network
{
    template<typename T, std::size_t InputSize_, std::size_t OutputSize_, std::size_t ParameterSize_>
    class Layer
    {
        static_assert(std::is_floating_point_v<T>, "Layer requires a floating-point type");

    public:
        using ValueType = T;
        using InputVector = math::Vector<T, InputSize_>;
        using OutputVector = math::Vector<T, OutputSize_>;
        using ParameterVector = math::Vector<T, ParameterSize_>;

        static constexpr std::size_t InputSize = InputSize_;
        static constexpr std::size_t OutputSize = OutputSize_;
        static constexpr std::size_t ParameterSize = ParameterSize_;

        virtual ~Layer() = default;

        virtual void Forward(const InputVector& input) = 0;
        virtual const InputVector& Backward(const OutputVector& outputGradient) = 0;
        virtual const OutputVector& Output() const = 0;
        virtual const ParameterVector& Parameters() const = 0;
        virtual void SetParameters(const ParameterVector& parameters) = 0;

    protected:
        Layer() = default;
        Layer(const Layer&) = default;
        Layer(Layer&&) = default;
        Layer& operator=(const Layer&) = default;
        Layer& operator=(Layer&&) = default;
    };
}
