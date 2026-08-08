neural_network_add_header_library(neural_network.losses)

target_include_directories(neural_network.losses ${NEURAL_NETWORK_VISIBILITY}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.losses ${NEURAL_NETWORK_VISIBILITY}
    infra.util
    numerical.math
    neural_network.activation
    numerical.optimization
    numerical.regularization
)

target_sources(neural_network.losses PRIVATE
    BinaryCrossEntropy.hpp
    CategoricalCrossEntropy.hpp
    Loss.hpp
    MeanAbsoluteError.hpp
    MeanSquaredError.hpp
)

neural_network_add_coverage_sources(neural_network.losses
    MeanSquaredError.cpp
    MeanAbsoluteError.cpp
    BinaryCrossEntropy.cpp
    CategoricalCrossEntropy.cpp
)

add_subdirectory(test)
