add_library(neural_network.simulator.shell INTERFACE)

target_include_directories(neural_network.simulator.shell INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_sources(neural_network.simulator.shell INTERFACE
    AppRunner.hpp
)

target_link_libraries(neural_network.simulator.shell INTERFACE
    ui.backend.qt
)
