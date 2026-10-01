add_executable(platform_qemu.canary_test)

target_link_libraries(platform_qemu.canary_test PRIVATE
    gmock_main
)

target_sources(platform_qemu.canary_test PRIVATE
    TestQemuCanary.cpp
)

neural_network_add_qemu_test(platform_qemu.canary_test EXPECT_FAILURE)
