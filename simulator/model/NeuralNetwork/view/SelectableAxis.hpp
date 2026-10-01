#pragma once

#include "ui/charts/AxisTransform.hpp"

namespace simulator::model::nn::view
{
    class SelectableAxis
        : public ui::charts::AxisTransform
    {
    public:
        explicit SelectableAxis(const ui::charts::AxisTransform& initial)
            : active{ &initial }
        {}

        void Select(const ui::charts::AxisTransform& axis)
        {
            active = &axis;
        }

        [[nodiscard]] float ToView(float value) const override
        {
            return active->ToView(value);
        }

        [[nodiscard]] float FromView(float view) const override
        {
            return active->FromView(view);
        }

        [[nodiscard]] bool IsPlottable(float value) const override
        {
            return active->IsPlottable(value);
        }

        [[nodiscard]] ui::charts::AxisRange RangeFor(std::span<const float> values) const override
        {
            return active->RangeFor(values);
        }

        [[nodiscard]] std::size_t Ticks(ui::charts::AxisRange range, std::span<ui::charts::Tick> out) const override
        {
            return active->Ticks(range, out);
        }

        [[nodiscard]] std::size_t GridLines(ui::charts::AxisRange range, std::span<ui::charts::GridLine> out) const override
        {
            return active->GridLines(range, out);
        }

        [[nodiscard]] std::string_view Title() const override
        {
            return active->Title();
        }

        [[nodiscard]] std::size_t FormatCursorValue(float value, std::span<char> out) const override
        {
            return active->FormatCursorValue(value, out);
        }

    private:
        const ui::charts::AxisTransform* active;
    };
}
