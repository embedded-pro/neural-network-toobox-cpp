add_library(neural_network.simulator.nn.view STATIC)

set_target_properties(neural_network.simulator.nn.view PROPERTIES AUTOMOC ON)

target_include_directories(neural_network.simulator.nn.view PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
)

target_link_libraries(neural_network.simulator.nn.view PUBLIC
    neural_network.simulator.nn.application
    neural_network.simulator.shell
    ui.charts
    ui.backend.qt
    Qt6::Widgets
)

target_sources(neural_network.simulator.nn.view PRIVATE
    NnMainWindow.hpp
    NnMainWindow.cpp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(neural_network.simulator.nn.view PRIVATE -Wno-error)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(neural_network.simulator.nn.view PRIVATE /W4 /wd4244 /wd4100)
endif()
