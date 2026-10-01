add_library(neural_network.simulator.model.neuralnetwork.view STATIC)

target_include_directories(neural_network.simulator.model.neuralnetwork.view PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.simulator.model.neuralnetwork.view PUBLIC
    neural_network.simulator.model.neuralnetwork.application
    neural_network.simulator.shell
    ui.charts
    ui.backend.qt
    Qt6::Widgets
)

set_target_properties(neural_network.simulator.model.neuralnetwork.view PROPERTIES
    AUTOMOC ON
)

target_sources(neural_network.simulator.model.neuralnetwork.view PRIVATE
    NnMainWindow.cpp
    NnMainWindow.hpp
    SelectableAxis.hpp
)

if(MSVC)
    target_compile_options(neural_network.simulator.model.neuralnetwork.view PRIVATE /W4 /wd4244 /wd4100)
else()
    target_compile_options(neural_network.simulator.model.neuralnetwork.view PRIVATE
        -Wno-error
    )
endif()
