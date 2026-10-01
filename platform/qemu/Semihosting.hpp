#pragma once

#include <cstdint>

namespace platform::qemu
{
    enum class SemihostingOperation : uint32_t
    {
        getCommandLine = 0x15u,
        exitExtended = 0x20u,
    };

    constexpr uint32_t applicationExit = 0x20026u;

    inline uint32_t SemihostingCall(SemihostingOperation operation, void* parameters)
    {
        uint32_t result;
        __asm volatile(
            "mov r0, %[op]\n"
            "mov r1, %[params]\n"
            "bkpt #0xAB\n"
            "mov %[result], r0\n"
            : [result] "=r"(result)
            : [op] "r"(static_cast<uint32_t>(operation)), [params] "r"(parameters)
            : "r0", "r1", "memory");
        return result;
    }

    [[noreturn]] inline void SemihostingExit(int status)
    {
        uint32_t block[2]{ applicationExit, static_cast<uint32_t>(status) };
        SemihostingCall(SemihostingOperation::exitExtended, block);

        while (true)
        {
        }
    }
}
