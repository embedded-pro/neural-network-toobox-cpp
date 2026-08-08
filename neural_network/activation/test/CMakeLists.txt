add_executable(neural_network.activation_test)
emil_build_for(neural_network.activation_test BOOL NEURAL_NETWORK_TOOLBOX_BUILD_TESTS)
emil_add_test(neural_network.activation_test)

target_link_libraries(neural_network.activation_test PUBLIC
    gmock_main
    neural_network.activation
)

target_sources(neural_network.activation_test PRIVATE
    TestLeakyReLU.cpp
    TestReLU.cpp
    TestSigmoid.cpp
    TestSoftmax.cpp
    TestTanh.cpp
)
