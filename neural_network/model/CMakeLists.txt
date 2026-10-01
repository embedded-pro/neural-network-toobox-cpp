neural_network_add_header_library(neural_network.model)

target_include_directories(neural_network.model ${NEURAL_NETWORK_VISIBILITY}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.model ${NEURAL_NETWORK_VISIBILITY}
    numerical.math
    neural_network.layer
    neural_network.losses
    numerical.optimization
)

target_sources(neural_network.model PRIVATE
    Model.hpp
)

neural_network_add_coverage_sources(neural_network.model
    Model.cpp
)

add_subdirectory(test)
