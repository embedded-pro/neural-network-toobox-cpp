function(neural_network_link_qemu_runtime target)
    if(NOT EMIL_BUILD_QEMU)
        return()
    endif()
    target_link_libraries(${target} PRIVATE
        hal.cortex_m
        hal.cortex_m.runtime
        hal.qemu.syscalls
        hal.qemu.default_init
        hal.qemu.sync
        hal.qemu.cortex
        gmock_main
    )
    if(TEST ${target})
        set_tests_properties(${target} PROPERTIES
            PASS_REGULAR_EXPRESSION "\\[  PASSED  \\] [1-9][0-9]* test"
            FAIL_REGULAR_EXPRESSION "\\[  FAILED  \\]"
            TIMEOUT 120
        )
    endif()
endfunction()
