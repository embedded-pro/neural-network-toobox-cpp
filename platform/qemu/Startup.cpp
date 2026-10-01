#include "Semihosting.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>

extern int main(int argc, char** argv);

extern "C"
{
    extern void Default_Handler_Forwarded();
    extern void HardwareInitialization();
    extern void __libc_init_array();
    extern void initialise_monitor_handles();

    static void Default_Handler();

    void Reset_Handler();
    void NMI_Handler() __attribute__((weak, alias("Default_Handler")));
    void HardFault_Handler() __attribute__((weak, alias("Default_Handler")));
    void MemManage_Handler() __attribute__((weak, alias("Default_Handler")));
    void BusFault_Handler() __attribute__((weak, alias("Default_Handler")));
    void UsageFault_Handler() __attribute__((weak, alias("Default_Handler")));
    void SVC_Handler() __attribute__((weak, alias("Default_Handler")));
    void DebugMon_Handler() __attribute__((weak, alias("Default_Handler")));
    void PendSV_Handler() __attribute__((weak, alias("Default_Handler")));
    void SysTick_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ0_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ1_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ2_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ3_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ4_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ5_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ6_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ7_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ8_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ9_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ10_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ11_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ12_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ13_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ14_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ15_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ16_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ17_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ18_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ19_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ20_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ21_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ22_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ23_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ24_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ25_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ26_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ27_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ28_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ29_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ30_Handler() __attribute__((weak, alias("Default_Handler")));
    void IRQ31_Handler() __attribute__((weak, alias("Default_Handler")));
}

extern "C" uint32_t _estack;
extern "C" uint32_t _sidata;
extern "C" uint32_t _sdata;
extern "C" uint32_t _edata;
extern "C" uint32_t _sbss;
extern "C" uint32_t _ebss;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
static void (*const g_pfnVectors[])(void) __attribute__((section(".isr_vector"), used)) = {
    reinterpret_cast<void (*)()>(&_estack),
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    SVC_Handler,
    DebugMon_Handler,
    nullptr,
    PendSV_Handler,
    SysTick_Handler,
    IRQ0_Handler,
    IRQ1_Handler,
    IRQ2_Handler,
    IRQ3_Handler,
    IRQ4_Handler,
    IRQ5_Handler,
    IRQ6_Handler,
    IRQ7_Handler,
    IRQ8_Handler,
    IRQ9_Handler,
    IRQ10_Handler,
    IRQ11_Handler,
    IRQ12_Handler,
    IRQ13_Handler,
    IRQ14_Handler,
    IRQ15_Handler,
    IRQ16_Handler,
    IRQ17_Handler,
    IRQ18_Handler,
    IRQ19_Handler,
    IRQ20_Handler,
    IRQ21_Handler,
    IRQ22_Handler,
    IRQ23_Handler,
    IRQ24_Handler,
    IRQ25_Handler,
    IRQ26_Handler,
    IRQ27_Handler,
    IRQ28_Handler,
    IRQ29_Handler,
    IRQ30_Handler,
    IRQ31_Handler,
};
#pragma GCC diagnostic pop

namespace
{
    constexpr std::size_t maxCommandLineLength = 512;
    constexpr std::size_t maxArguments = 16;

    std::array<char, maxCommandLineLength> commandLine{};
    std::array<char*, maxArguments + 1> arguments{};

    int ParseCommandLine()
    {
        struct
        {
            char* buffer;
            uint32_t length;
        } block{ commandLine.data(), static_cast<uint32_t>(commandLine.size() - 1) };

        if (platform::qemu::SemihostingCall(platform::qemu::SemihostingOperation::getCommandLine, &block) != 0u)
            block.length = 0;

        commandLine[block.length] = '\0';

        int count = 0;
        bool inArgument = false;
        for (uint32_t i = 0; i < block.length; ++i)
        {
            if (commandLine[i] == ' ')
            {
                commandLine[i] = '\0';
                inArgument = false;
            }
            else if (!inArgument && count < static_cast<int>(maxArguments))
            {
                arguments[count++] = &commandLine[i];
                inArgument = true;
            }
        }

        arguments[count] = nullptr;
        return count;
    }
}

extern "C" void Reset_Handler()
{
    __asm volatile("cpsid i");

    *reinterpret_cast<volatile uint32_t*>(0xE000ED88U) |= (0xFU << 20U);
    __asm volatile("dsb" ::: "memory");
    __asm volatile("isb" ::: "memory");

    const uint32_t* src = &_sidata;
    for (uint32_t* dst = &_sdata; dst < &_edata;)
        *dst++ = *src++;

    for (uint32_t* bss = &_sbss; bss < &_ebss;)
        *bss++ = 0U;

    initialise_monitor_handles();
    __libc_init_array();

    __asm volatile("cpsie i");

    HardwareInitialization();

    const int argc = ParseCommandLine();
    std::exit(main(argc, arguments.data()));
}

static void Default_Handler()
{
    Default_Handler_Forwarded();
}
