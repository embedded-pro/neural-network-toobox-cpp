add_executable(neural_network.losses_test)
emil_build_for(neural_network.losses_test BOOL NEURAL_NETWORK_TOOLBOX_BUILD_TESTS)
emil_add_test(neural_network.losses_test)

target_link_libraries(neural_network.losses_test PUBLIC
    gmock_main
    neural_network.losses
)

neural_network_link_qemu_runtime(neural_network.losses_test)

target_sources(neural_network.losses_test PRIVATE
    LossTestSupport.hpp
    TestBinaryCrossEntropy.cpp
    TestCategoricalCrossEntropy.cpp
    TestMeanAbsoluteError.cpp
    TestMeanSquaredError.cpp
)
