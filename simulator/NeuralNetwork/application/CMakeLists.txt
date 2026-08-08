add_library(neural_network.simulator.nn.application STATIC)

target_include_directories(neural_network.simulator.nn.application PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
)

target_link_libraries(neural_network.simulator.nn.application PUBLIC
    neural_network.model
    numerical.regularization
)

target_sources(neural_network.simulator.nn.application PRIVATE
    NnSimulator.hpp
    NnSimulator.cpp
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(neural_network.simulator.nn.application PRIVATE -Wno-error)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(neural_network.simulator.nn.application PRIVATE /W4 /wd4244 /wd4100)
endif()
