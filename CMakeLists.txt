if (APPLE)
    cmake_minimum_required(VERSION 3.3.2)
else (APPLE)
    cmake_minimum_required(VERSION 2.8)
endif()

project(MWAC_HIRES)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    # cuda library
    cuda_add_library(cuda_hires SHARED ${CMAKE_SOURCE_DIR}/cuda/cuda_hires.cu)
    cuda_add_cufft_to_target(cuda_hires)
    # other library
    #
    # ....
    #
    # executable
    add_executable(mwac_hires ${CMAKE_SOURCE_DIR}/main/mwac_hires.c)
    include_directories(${CMAKE_SOURCE_DIR}/cuda)
    target_link_libraries(mwac_hires cuda_hires)

    # dependencies
    add_dependencies(mwac_hires cuda_hires)



    # install
    install(FILES ${CMAKE_SOURCE_DIR}/cuda/cuda_hires.h DESTINATION include)
    install(TARGETS cuda_hires LIBRARY DESTINATION lib)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/mwac_hires DESTINATION bin)

endif()
