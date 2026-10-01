if(NOT TARGET platform_qemu_startup)
    add_library(platform_qemu_startup STATIC
        DefaultInit.cpp
        FaultHandler.cpp
        FaultHandler.hpp
        Semihosting.hpp
        Startup.cpp
    )

    set_target_properties(platform_qemu_startup PROPERTIES COMPILE_WARNING_AS_ERROR Off)

    target_include_directories(platform_qemu_startup PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_options(platform_qemu_startup INTERFACE
        -T ${CMAKE_CURRENT_SOURCE_DIR}/mps2.ld
        -Wl,-u,Reset_Handler
        -Wl,-u,FaultDispatch
        -Wl,-u,main
        -Wl,-u,_printf_float
    )

    target_link_libraries(platform_qemu_startup INTERFACE
        -lrdimon
    )
endif()

if(NEURAL_NETWORK_TOOLBOX_BUILD_TESTS)
    add_subdirectory(test)
endif()
