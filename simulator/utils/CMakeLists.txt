add_library(neural_network.simulator.utils STATIC)

target_include_directories(neural_network.simulator.utils PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.simulator.utils PUBLIC
    numerical.analysis
)

target_sources(neural_network.simulator.utils PRIVATE
    SignalGenerator.cpp
    SignalGenerator.hpp
    TwiddleFactorsTable.hpp
)

if(MSVC)
    target_compile_options(neural_network.simulator.utils PRIVATE /W4 /wd4244 /wd4100)
else()
    target_compile_options(neural_network.simulator.utils PRIVATE
        -Wno-error
    )
endif()
