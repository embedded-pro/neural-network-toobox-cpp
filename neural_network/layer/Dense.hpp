#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "infra/util/ReallyAssert.hpp"
#include "neural_network/activation/ActivationFunction.hpp"
#include "neural_network/layer/Layer.hpp"
#include "numerical/math/CompilerOptimizations.hpp"
#include "numerical/math/Matrix.hpp"

namespace neural_network
{
    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    class Dense
        : public Layer<T, InputSize, OutputSize, (InputSize * OutputSize) + OutputSize>
    {
        static_assert(std::is_floating_point_v<T>, "Dense requires a floating-point type");

    public:
        using BaseLayer = Layer<T, InputSize, OutputSize, (InputSize * OutputSize) + OutputSize>;
        using InputVector = typename BaseLayer::InputVector;
        using OutputVector = typename BaseLayer::OutputVector;
        using ParameterVector = typename BaseLayer::ParameterVector;
        using WeightMatrix = math::Matrix<T, OutputSize, InputSize>;

        Dense(const WeightMatrix& initialWeights, const ActivationFunction<T>& activationFunction);
        Dense(const WeightMatrix& initialWeights, const ActivationFunction<T>&& activationFunction) = delete;

        void Forward(const InputVector& layerInput) override;
        const InputVector& Backward(const OutputVector& outputGradient) override;
        const OutputVector& Output() const override;
        const ParameterVector& Parameters() const override;
        void SetParameters(const ParameterVector& newParameters) override;

    private:
        static constexpr std::size_t biasOffset = InputSize * OutputSize;

        T& Weight(std::size_t row, std::size_t column);
        const T& Weight(std::size_t row, std::size_t column) const;
        const T& Bias(std::size_t row) const;

        const ActivationFunction<T>& activation;

        ParameterVector parameters;
        ParameterVector parameterGradients;

        InputVector input;
        OutputVector preActivation;
        OutputVector output;
        InputVector inputGradient;
        bool forwardDone{ false };
    };

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    Dense<T, InputSize, OutputSize>::Dense(const WeightMatrix& initialWeights, const ActivationFunction<T>& activationFunction)
        : activation{ activationFunction }
    {
        for (std::size_t i = 0; i < OutputSize; ++i)
            for (std::size_t j = 0; j < InputSize; ++j)
                Weight(i, j) = initialWeights.at(i, j);
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    OPTIMIZE_FOR_SPEED void Dense<T, InputSize, OutputSize>::Forward(const InputVector& layerInput)
    {
        input = layerInput;

        for (std::size_t i = 0; i < OutputSize; ++i)
        {
            T sum{ Bias(i) };
            for (std::size_t j = 0; j < InputSize; ++j)
                sum += Weight(i, j) * input[j];
            preActivation[i] = sum;
        }

        activation.ForwardVector(output, preActivation);
        forwardDone = true;
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    OPTIMIZE_FOR_SPEED const typename Dense<T, InputSize, OutputSize>::InputVector& Dense<T, InputSize, OutputSize>::Backward(const OutputVector& outputGradient)
    {
        really_assert(forwardDone);

        OutputVector preActivationGradient{};
        activation.BackwardVector(preActivationGradient, preActivation, output, outputGradient);

        for (std::size_t j = 0; j < InputSize; ++j)
        {
            T sum{ 0 };
            for (std::size_t i = 0; i < OutputSize; ++i)
                sum += Weight(i, j) * preActivationGradient[i];
            inputGradient[j] = sum;
        }

        for (std::size_t i = 0; i < OutputSize; ++i)
        {
            for (std::size_t j = 0; j < InputSize; ++j)
                parameterGradients[i * InputSize + j] = preActivationGradient[i] * input[j];

            parameterGradients[biasOffset + i] = preActivationGradient[i];
        }

        return inputGradient;
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    const typename Dense<T, InputSize, OutputSize>::OutputVector& Dense<T, InputSize, OutputSize>::Output() const
    {
        return output;
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    const typename Dense<T, InputSize, OutputSize>::ParameterVector& Dense<T, InputSize, OutputSize>::Parameters() const
    {
        return parameters;
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    void Dense<T, InputSize, OutputSize>::SetParameters(const ParameterVector& newParameters)
    {
        parameters = newParameters;
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    T& Dense<T, InputSize, OutputSize>::Weight(std::size_t row, std::size_t column)
    {
        return parameters[row * InputSize + column];
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    const T& Dense<T, InputSize, OutputSize>::Weight(std::size_t row, std::size_t column) const
    {
        return parameters[row * InputSize + column];
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize>
    const T& Dense<T, InputSize, OutputSize>::Bias(std::size_t row) const
    {
        return parameters[biasOffset + row];
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class Dense<float, 3, 2>;
#endif
}
