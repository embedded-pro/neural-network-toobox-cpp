add_executable(neural_network.layer_test)
emil_build_for(neural_network.layer_test BOOL NEURAL_NETWORK_TOOLBOX_BUILD_TESTS)
emil_add_test(neural_network.layer_test)

target_link_libraries(neural_network.layer_test PUBLIC
    gmock_main
    neural_network.activation
    neural_network.layer
)

neural_network_add_qemu_test(neural_network.layer_test)

target_sources(neural_network.layer_test PRIVATE
    TestDense.cpp
)
