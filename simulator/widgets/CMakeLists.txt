find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_library(neural_network.simulator.widgets STATIC)

target_include_directories(neural_network.simulator.widgets PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.simulator.widgets PUBLIC
    neural_network.simulator.utils
    Qt6::Widgets
)

set_target_properties(neural_network.simulator.widgets PROPERTIES
    AUTOMOC ON
)

target_sources(neural_network.simulator.widgets PRIVATE
    ChartInteraction.cpp
    ChartInteraction.hpp
    FrequencyChartWidget.cpp
    FrequencyChartWidget.hpp
    SignalConfigWidget.cpp
    SignalConfigWidget.hpp
    TimeSeriesChartWidget.cpp
    TimeSeriesChartWidget.hpp
)

if(MSVC)
    target_compile_options(neural_network.simulator.widgets PRIVATE /W4 /wd4244 /wd4100)
else()
    target_compile_options(neural_network.simulator.widgets PRIVATE
        -Wno-error
    )
endif()
