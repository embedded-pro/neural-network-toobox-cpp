#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3", "fast-math")
#endif

#include "neural_network/layer/Dense.hpp"
#include "neural_network/layer/Layer.hpp"
#include "neural_network/losses/Loss.hpp"
#include "numerical/math/CompilerOptimizations.hpp"
#include "numerical/math/Matrix.hpp"
#include "numerical/optimization/Optimizer.hpp"
#include <array>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace neural_network
{
    template<typename LayerType, typename... Args>
    auto make_layer(Args&&... args)
    {
        return [arguments = std::tuple<Args...>{ std::forward<Args>(args)... }]() mutable
        {
            return std::make_from_tuple<LayerType>(std::move(arguments));
        };
    }

    namespace detail
    {
        template<typename T, typename L>
        inline constexpr bool is_layer_v = std::is_base_of_v<Layer<T, L::InputSize, L::OutputSize, L::ParameterSize>, L>;

        template<std::size_t InputSize, std::size_t OutputSize, typename... Layers>
        constexpr bool LayerSizesChain()
        {
            if constexpr (sizeof...(Layers) == 0)
                return false;
            else
            {
                constexpr std::array<std::size_t, sizeof...(Layers)> inputSizes{ Layers::InputSize... };
                constexpr std::array<std::size_t, sizeof...(Layers)> outputSizes{ Layers::OutputSize... };

                if (inputSizes.front() != InputSize || outputSizes.back() != OutputSize)
                    return false;

                for (std::size_t i = 1; i < sizeof...(Layers); ++i)
                    if (inputSizes[i] != outputSizes[i - 1])
                        return false;

                return true;
            }
        }
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    class Model
    {
        static_assert(std::is_floating_point_v<T>, "Model requires a floating-point type");
        static_assert(sizeof...(Layers) > 0, "Model must have at least one layer");
        static_assert((std::is_same_v<typename Layers::ValueType, T> && ...), "All layers must share the Model value type");
        static_assert((detail::is_layer_v<T, Layers> && ...), "All types in Layers must derive from Layer");
        static_assert(detail::LayerSizesChain<InputSize, OutputSize, Layers...>(), "Layer sizes do not match");

    public:
        static constexpr std::size_t TotalParameters = (Layers::ParameterSize + ...);

        using InputVector = math::Vector<T, InputSize>;
        using OutputVector = math::Vector<T, OutputSize>;
        using ParameterVector = math::Vector<T, TotalParameters>;

        Model()
        requires(std::is_default_constructible_v<Layers> && ...)
        = default;

        template<typename... FactoryFuncs>
        requires(sizeof...(FactoryFuncs) == sizeof...(Layers)) && (std::is_invocable_r_v<Layers, FactoryFuncs> && ...)
        explicit Model(FactoryFuncs&&... factories)
            : layers{ std::invoke(std::forward<FactoryFuncs>(factories))... }
        {}

        OutputVector Forward(const InputVector& input);
        InputVector Backward(const OutputVector& outputGradient);
        void Train(optimization::Optimizer<T, TotalParameters>& optimizer, Loss<T, TotalParameters>& loss, const ParameterVector& initialParameters);
        void SetParameters(const ParameterVector& parameters);
        ParameterVector GetParameters() const;

    private:
        template<std::size_t... Is>
        void ForwardImpl(const InputVector& input, std::index_sequence<Is...>);

        template<std::size_t I>
        void ForwardLayer(const InputVector& input);

        template<std::size_t I, typename GradientType>
        InputVector BackwardLayer(const GradientType& gradient);

        template<std::size_t... Is>
        void SetParametersImpl(const ParameterVector& parameters, std::index_sequence<Is...>);

        template<typename LayerType>
        void SetLayerParameters(LayerType& layer, const ParameterVector& parameters, std::size_t& offset);

        template<std::size_t... Is>
        ParameterVector GetParametersImpl(std::index_sequence<Is...>) const;

        template<typename LayerType>
        void GetLayerParameters(const LayerType& layer, ParameterVector& parameters, std::size_t& offset) const;

        std::tuple<Layers...> layers;
    };

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    OPTIMIZE_FOR_SPEED typename Model<T, InputSize, OutputSize, Layers...>::OutputVector Model<T, InputSize, OutputSize, Layers...>::Forward(const InputVector& input)
    {
        ForwardImpl(input, std::make_index_sequence<sizeof...(Layers)>{});
        return std::get<sizeof...(Layers) - 1>(layers).Output();
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    OPTIMIZE_FOR_SPEED typename Model<T, InputSize, OutputSize, Layers...>::InputVector Model<T, InputSize, OutputSize, Layers...>::Backward(const OutputVector& outputGradient)
    {
        return BackwardLayer<sizeof...(Layers) - 1>(outputGradient);
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    void Model<T, InputSize, OutputSize, Layers...>::Train(optimization::Optimizer<T, TotalParameters>& optimizer, Loss<T, TotalParameters>& loss, const ParameterVector& initialParameters)
    {
        const auto& result{ optimizer.Minimize(initialParameters, loss) };
        SetParameters(result.parameters);
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    void Model<T, InputSize, OutputSize, Layers...>::SetParameters(const ParameterVector& parameters)
    {
        SetParametersImpl(parameters, std::make_index_sequence<sizeof...(Layers)>{});
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    typename Model<T, InputSize, OutputSize, Layers...>::ParameterVector Model<T, InputSize, OutputSize, Layers...>::GetParameters() const
    {
        return GetParametersImpl(std::make_index_sequence<sizeof...(Layers)>{});
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    template<std::size_t... Is>
    void Model<T, InputSize, OutputSize, Layers...>::ForwardImpl(const InputVector& input, std::index_sequence<Is...>)
    {
        (ForwardLayer<Is>(input), ...);
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    template<std::size_t I>
    void Model<T, InputSize, OutputSize, Layers...>::ForwardLayer(const InputVector& input)
    {
        if constexpr (I == 0)
            std::get<0>(layers).Forward(input);
        else
            std::get<I>(layers).Forward(std::get<I - 1>(layers).Output());
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    template<std::size_t I, typename GradientType>
    typename Model<T, InputSize, OutputSize, Layers...>::InputVector Model<T, InputSize, OutputSize, Layers...>::BackwardLayer(const GradientType& gradient)
    {
        const auto& inputGradient{ std::get<I>(layers).Backward(gradient) };

        if constexpr (I == 0)
            return inputGradient;
        else
            return BackwardLayer<I - 1>(inputGradient);
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    template<std::size_t... Is>
    void Model<T, InputSize, OutputSize, Layers...>::SetParametersImpl(const ParameterVector& parameters, std::index_sequence<Is...>)
    {
        std::size_t offset{ 0 };
        (SetLayerParameters(std::get<Is>(layers), parameters, offset), ...);
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    template<typename LayerType>
    void Model<T, InputSize, OutputSize, Layers...>::SetLayerParameters(LayerType& layer, const ParameterVector& parameters, std::size_t& offset)
    {
        typename LayerType::ParameterVector layerParameters{};

        for (std::size_t i = 0; i < LayerType::ParameterSize; ++i)
            layerParameters[i] = parameters[offset + i];

        layer.SetParameters(layerParameters);
        offset += LayerType::ParameterSize;
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    template<std::size_t... Is>
    typename Model<T, InputSize, OutputSize, Layers...>::ParameterVector Model<T, InputSize, OutputSize, Layers...>::GetParametersImpl(std::index_sequence<Is...>) const
    {
        ParameterVector parameters{};
        std::size_t offset{ 0 };
        (GetLayerParameters(std::get<Is>(layers), parameters, offset), ...);
        return parameters;
    }

    template<typename T, std::size_t InputSize, std::size_t OutputSize, typename... Layers>
    template<typename LayerType>
    void Model<T, InputSize, OutputSize, Layers...>::GetLayerParameters(const LayerType& layer, ParameterVector& parameters, std::size_t& offset) const
    {
        const auto& layerParameters{ layer.Parameters() };

        for (std::size_t i = 0; i < LayerType::ParameterSize; ++i)
            parameters[offset + i] = layerParameters[i];

        offset += LayerType::ParameterSize;
    }

#ifdef NEURAL_NETWORK_TOOLBOX_COVERAGE_BUILD
    extern template class Model<float, 2, 1, Dense<float, 2, 3>, Dense<float, 3, 1>>;
#endif
}
