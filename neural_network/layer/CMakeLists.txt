neural_network_add_header_library(neural_network.layer)

target_include_directories(neural_network.layer ${NEURAL_NETWORK_VISIBILITY}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.layer ${NEURAL_NETWORK_VISIBILITY}
    infra.util
    numerical.math
)

target_sources(neural_network.layer PRIVATE
    Dense.hpp
    Layer.hpp
)

neural_network_add_coverage_sources(neural_network.layer
    Dense.cpp
)

add_subdirectory(test)
