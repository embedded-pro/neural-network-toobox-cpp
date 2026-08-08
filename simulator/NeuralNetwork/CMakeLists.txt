find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_subdirectory(application)
add_subdirectory(view)

add_executable(neural_network.simulator.nn)
emil_build_for(neural_network.simulator.nn HOST All PREREQUISITE_BOOL NEURAL_NETWORK_TOOLBOX_BUILD_SIMULATOR)

target_link_libraries(neural_network.simulator.nn PRIVATE
    neural_network.simulator.nn.application
    neural_network.simulator.nn.view
)

target_sources(neural_network.simulator.nn PRIVATE
    Main.cpp
)

if(MSVC)
    target_compile_options(neural_network.simulator.nn PRIVATE /W4 /wd4244 /wd4100)
else()
    target_compile_options(neural_network.simulator.nn PRIVATE
        -Wno-error
    )
endif()
