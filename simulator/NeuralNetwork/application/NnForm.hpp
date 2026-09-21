#pragma once

#include "simulator/NeuralNetwork/application/NnSimulator.hpp"
#include "ui/model/FormModel.hpp"
#include <array>

namespace simulator::neural_network::nn
{
    namespace field
    {
        inline constexpr ui::model::FieldId demo{ 1 };
        inline constexpr ui::model::FieldId hiddenSize{ 2 };
        inline constexpr ui::model::FieldId learningRate{ 3 };
        inline constexpr ui::model::FieldId epochs{ 4 };
        inline constexpr ui::model::FieldId sineSamples{ 5 };

        inline constexpr ui::model::GroupId selection{ 1 };
        inline constexpr ui::model::GroupId network{ 2 };

        inline constexpr ui::model::ActionId train{ 1 };
    }

    class NnForm
    {
    public:
        NnForm();

        [[nodiscard]] ui::model::FormModel& Model();
        [[nodiscard]] const ui::model::FormModel& Model() const;

        [[nodiscard]] NnSimulator::Configuration BuildConfiguration() const;

    private:
        static constexpr std::size_t fieldCount{ 5 };

        std::array<ui::model::FieldValue, fieldCount> values{};

        ui::model::FormSpec spec;
        ui::model::FormModel model;
    };
}
