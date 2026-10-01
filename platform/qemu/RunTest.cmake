foreach(required QEMU QEMU_MACHINE QEMU_KERNEL QEMU_EXPECTATION)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "RunTest.cmake: ${required} is not set")
    endif()
endforeach()

set(command ${QEMU}
    -machine ${QEMU_MACHINE}
    -nographic
    -no-reboot
    -semihosting-config enable=on,target=native
    -kernel ${QEMU_KERNEL}
)

if(DEFINED QEMU_ARGUMENTS AND NOT QEMU_ARGUMENTS STREQUAL "")
    list(APPEND command -append "${QEMU_ARGUMENTS}")
endif()

execute_process(
    COMMAND ${command}
    RESULT_VARIABLE exitCode
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    ECHO_OUTPUT_VARIABLE
    ECHO_ERROR_VARIABLE
    TIMEOUT 110
)

message(STATUS "qemu exit code: ${exitCode}")

if(NOT exitCode MATCHES "^[0-9]+$")
    message(FATAL_ERROR "qemu did not exit normally: ${exitCode}")
endif()

string(REGEX MATCH "\\[  PASSED  \\] ([0-9]+) tests?\\." passedLine "${output}")
set(passed "${CMAKE_MATCH_1}")
if(passed STREQUAL "" OR passed EQUAL 0)
    message(FATAL_ERROR "no passing gtest test was reported; test registrations or console output are missing")
endif()

string(REGEX MATCH "\\[  FAILED  \\] ([0-9]+) tests?, listed below" failedLine "${output}")
set(failed "${CMAKE_MATCH_1}")

if(QEMU_EXPECTATION STREQUAL "success")
    if(NOT exitCode EQUAL 0 OR NOT failed STREQUAL "")
        message(FATAL_ERROR "gtest reported failures (exit code ${exitCode})")
    endif()
elseif(QEMU_EXPECTATION STREQUAL "failure")
    if(exitCode EQUAL 0 OR NOT failed EQUAL 1)
        message(FATAL_ERROR "the canary failure was not propagated (exit code ${exitCode}, failed tests '${failed}')")
    endif()
else()
    message(FATAL_ERROR "RunTest.cmake: unknown QEMU_EXPECTATION '${QEMU_EXPECTATION}'")
endif()
