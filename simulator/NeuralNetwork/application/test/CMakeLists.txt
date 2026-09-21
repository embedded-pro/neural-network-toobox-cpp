add_executable(neural_network.simulator.nn.application_test)
emil_build_for(neural_network.simulator.nn.application_test HOST All PREREQUISITE_BOOL NEURAL_NETWORK_TOOLBOX_BUILD_SIMULATOR)
emil_add_test(neural_network.simulator.nn.application_test)

target_link_libraries(neural_network.simulator.nn.application_test PUBLIC
    gmock_main
    neural_network.simulator.nn.application
)

target_sources(neural_network.simulator.nn.application_test PRIVATE
    TestNnForm.cpp
)
