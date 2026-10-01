set(NEURAL_NETWORK_QEMU_RUN_TEST_SCRIPT ${CMAKE_CURRENT_LIST_DIR}/../platform/qemu/RunTest.cmake)

function(neural_network_add_qemu_test target)
    if(NOT DEFINED QEMU_MACHINE)
        return()
    endif()

    cmake_parse_arguments(PARSE_ARGV 1 QEMU_TEST "EXPECT_FAILURE" "" "")

    find_program(QEMU_SYSTEM_ARM qemu-system-arm REQUIRED)

    if(QEMU_TEST_EXPECT_FAILURE)
        set(expectation "failure")
    else()
        set(expectation "success")
    endif()

    target_link_libraries(${target} PRIVATE platform_qemu_startup)

    add_test(
        NAME qemu.${target}
        COMMAND ${CMAKE_COMMAND}
            -DQEMU=${QEMU_SYSTEM_ARM}
            -DQEMU_MACHINE=${QEMU_MACHINE}
            -DQEMU_KERNEL=$<TARGET_FILE:${target}>
            -DQEMU_EXPECTATION=${expectation}
            -P ${NEURAL_NETWORK_QEMU_RUN_TEST_SCRIPT}
    )
    set_tests_properties(qemu.${target} PROPERTIES TIMEOUT 120)
endfunction()
