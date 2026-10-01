add_library(neural_network.simulator.model.neuralnetwork.application STATIC)

target_include_directories(neural_network.simulator.model.neuralnetwork.application PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.simulator.model.neuralnetwork.application PUBLIC
    ui.model
)

target_sources(neural_network.simulator.model.neuralnetwork.application PRIVATE
    Mlp.cpp
    Mlp.hpp
    NnSimulator.cpp
    NnSimulator.hpp
    NnForm.cpp
    NnForm.hpp
)

if(MSVC)
    target_compile_options(neural_network.simulator.model.neuralnetwork.application PRIVATE /W4 /wd4244 /wd4100)
else()
    target_compile_options(neural_network.simulator.model.neuralnetwork.application PRIVATE
        -Wno-error
    )
endif()

if (NEURAL_NETWORK_TOOLBOX_BUILD_TESTS)
    add_subdirectory(test)
endif()
