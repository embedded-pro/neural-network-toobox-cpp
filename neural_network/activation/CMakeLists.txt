neural_network_add_header_library(neural_network.activation)

target_include_directories(neural_network.activation ${NEURAL_NETWORK_VISIBILITY}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(neural_network.activation ${NEURAL_NETWORK_VISIBILITY}
    infra.util
    numerical.math
)

target_sources(neural_network.activation PRIVATE
    ActivationFunction.hpp
    LeakyReLU.hpp
    ReLU.hpp
    Sigmoid.hpp
    Softmax.hpp
    Tanh.hpp
)

neural_network_add_coverage_sources(neural_network.activation
    ActivationFunction.cpp
    ReLU.cpp
    LeakyReLU.cpp
    Sigmoid.cpp
    Tanh.cpp
    Softmax.cpp
)

add_subdirectory(test)
