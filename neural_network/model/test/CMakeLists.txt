add_executable(neural_network.model_test)
emil_build_for(neural_network.model_test BOOL NEURAL_NETWORK_TOOLBOX_BUILD_TESTS)
emil_add_test(neural_network.model_test)

target_link_libraries(neural_network.model_test PUBLIC
    gmock_main
    neural_network.model
)

target_sources(neural_network.model_test PRIVATE
    TestModel.cpp
)
