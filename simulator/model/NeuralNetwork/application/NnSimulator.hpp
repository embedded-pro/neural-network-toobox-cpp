#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace simulator::model::nn
{
    enum class DemoType : std::int64_t
    {
        Xor = 1,
        SineApproximation = 2
    };

    struct NnConfig
    {
        DemoType demo{ DemoType::Xor };
        std::size_t hiddenSize{ 8 };
        float learningRate{ 5.0f };
        std::size_t epochs{ 5000 };
        std::size_t sineSamples{ 50 };
    };

    struct NnResult
    {
        std::vector<float> epochIndex;
        std::vector<float> lossHistory;
        std::vector<float> predictionAxis;
        std::vector<float> predictionTargets;
        std::vector<float> predictionOutputs;
        float finalLoss{ 0.0f };
    };

    class NnSimulator
    {
    public:
        struct Configuration
        {
            NnConfig nn;
        };

        static constexpr std::uint32_t seed{ 42 };
        static constexpr std::size_t minimumSineSamples{ 2 };
        static constexpr std::size_t minimumHiddenSize{ 1 };

        void Configure(const Configuration& config);
        [[nodiscard]] std::optional<NnResult> Run() const;

    private:
        Configuration configuration;
    };
}
