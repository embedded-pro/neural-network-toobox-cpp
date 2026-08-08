add_executable(neural_network.losses_test)
emil_build_for(neural_network.losses_test BOOL NEURAL_NETWORK_TOOLBOX_BUILD_TESTS)
emil_add_test(neural_network.losses_test)

target_link_libraries(neural_network.losses_test PUBLIC
    gmock_main
    neural_network.losses
)

target_sources(neural_network.losses_test PRIVATE
    TestBinaryCrossEntropy.cpp
    TestCategoricalCrossEntropy.cpp
    TestMeanAbsoluteError.cpp
    TestMeanSquaredError.cpp
)
