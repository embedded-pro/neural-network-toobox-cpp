find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_subdirectory(application)
add_subdirectory(view)

add_executable(neural_network.simulator.model.neural_network)
emil_build_for(neural_network.simulator.model.neural_network HOST All PREREQUISITE_BOOL NEURAL_NETWORK_TOOLBOX_BUILD_SIMULATOR)

target_link_libraries(neural_network.simulator.model.neural_network PRIVATE
    neural_network.simulator.model.neuralnetwork.application
    neural_network.simulator.model.neuralnetwork.view
    neural_network.simulator.shell
)

target_sources(neural_network.simulator.model.neural_network PRIVATE
    Main.cpp
)

if(MSVC)
    target_compile_options(neural_network.simulator.model.neural_network PRIVATE /W4 /wd4244 /wd4100)
else()
    target_compile_options(neural_network.simulator.model.neural_network PRIVATE
        -Wno-error
    )
endif()
