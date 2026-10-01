add_executable(neural_network.simulator.model.neuralnetwork.application_test)
emil_build_for(neural_network.simulator.model.neuralnetwork.application_test HOST All PREREQUISITE_BOOL NEURAL_NETWORK_TOOLBOX_BUILD_SIMULATOR)
emil_add_test(neural_network.simulator.model.neuralnetwork.application_test)

target_link_libraries(neural_network.simulator.model.neuralnetwork.application_test PUBLIC
    gmock_main
    neural_network.simulator.model.neuralnetwork.application
)

target_sources(neural_network.simulator.model.neuralnetwork.application_test PRIVATE
    TestMlp.cpp
    TestNnForm.cpp
    TestNnSimulator.cpp
)
