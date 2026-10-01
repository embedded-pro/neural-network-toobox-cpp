#include "simulator/model/NeuralNetwork/application/NnSimulator.hpp"
#include "simulator/model/NeuralNetwork/application/Mlp.hpp"
#include <cmath>
#include <numbers>

namespace simulator::model::nn
{
    namespace
    {
        struct TrainingData
        {
            std::vector<std::vector<float>> inputs;
            std::vector<float> targets;
            std::vector<float> axis;
        };

        TrainingData GenerateXorData()
        {
            return TrainingData{
                { { 0.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f } },
                { 0.0f, 1.0f, 1.0f, 0.0f },
                { 0.0f, 1.0f, 2.0f, 3.0f }
            };
        }

        TrainingData GenerateSineData(std::size_t samples)
        {
            TrainingData data;
            data.inputs.resize(samples);
            data.targets.resize(samples);
            data.axis.resize(samples);

            for (std::size_t i = 0; i < samples; ++i)
            {
                const auto x{ static_cast<float>(i) / static_cast<float>(samples - 1) };
                data.inputs[i] = { x };
                data.targets[i] = (std::sin(2.0f * std::numbers::pi_v<float> * x) + 1.0f) / 2.0f;
                data.axis[i] = x;
            }

            return data;
        }

        float TrainEpoch(Mlp& network, const TrainingData& data, float learningRate)
        {
            auto squaredError{ 0.0f };

            for (std::size_t s = 0; s < data.inputs.size(); ++s)
                squaredError += network.Accumulate(data.inputs[s], data.targets[s]);

            network.Step(learningRate, data.inputs.size());

            return squaredError / static_cast<float>(data.inputs.size());
        }

        void Predict(Mlp& network, const TrainingData& data, NnResult& result)
        {
            result.predictionAxis = data.axis;
            result.predictionTargets = data.targets;
            result.predictionOutputs.resize(data.inputs.size());

            auto squaredError{ 0.0f };

            for (std::size_t s = 0; s < data.inputs.size(); ++s)
            {
                result.predictionOutputs[s] = network.Predict(data.inputs[s]);
                const auto error{ result.predictionOutputs[s] - data.targets[s] };
                squaredError += error * error;
            }

            result.finalLoss = squaredError / static_cast<float>(data.inputs.size());
        }

        NnResult TrainNetwork(const NnConfig& config, const TrainingData& data)
        {
            Mlp network{ data.inputs.front().size(), config.hiddenSize, NnSimulator::seed };

            NnResult result;
            result.epochIndex.resize(config.epochs);
            result.lossHistory.resize(config.epochs);

            for (std::size_t epoch = 0; epoch < config.epochs; ++epoch)
            {
                result.epochIndex[epoch] = static_cast<float>(epoch);
                result.lossHistory[epoch] = TrainEpoch(network, data, config.learningRate);
            }

            Predict(network, data, result);

            return result;
        }
    }

    void NnSimulator::Configure(const Configuration& config)
    {
        configuration = config;
    }

    std::optional<NnResult> NnSimulator::Run() const
    {
        const auto& nn{ configuration.nn };

        if (nn.hiddenSize < minimumHiddenSize)
            return std::nullopt;

        switch (nn.demo)
        {
            case DemoType::Xor:
                return TrainNetwork(nn, GenerateXorData());
            case DemoType::SineApproximation:
                if (nn.sineSamples < minimumSineSamples)
                    return std::nullopt;
                return TrainNetwork(nn, GenerateSineData(nn.sineSamples));
        }

        return std::nullopt;
    }
}
