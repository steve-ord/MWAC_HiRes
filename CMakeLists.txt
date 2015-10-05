if (APPLE)
    cmake_minimum_required(VERSION 3.3.2)
else (APPLE)
    cmake_minimum_required(VERSION 2.8)
endif()

project(MWAC_HIRES)

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    # cuda library
    cuda_include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/ )
    cuda_add_library(cuda_hires SHARED ${CMAKE_SOURCE_DIR}/cuda/cuda_hires.cu)
    cuda_add_cufft_to_target(cuda_hires)
    # other library
    #
    # ....
    #
    # executable
    include_directories(${CMAKE_SOURCE_DIR}/cuda)
    add_executable(mwac_hires ${CMAKE_SOURCE_DIR}/main/mwac_hires.c)
    add_executable(test_cufft ${CMAKE_SOURCE_DIR}/test/test_cufft.c)
    add_executable(test_mkfile ${CMAKE_SOURCE_DIR}/test/test_mkfile.c)

    # dependencies
    add_dependencies(mwac_hires cuda_hires)
    add_dependencies(test_cufft cuda_hires)
    add_dependencies(test_mkfile cuda_hires)

    # linking
    target_link_libraries(mwac_hires cuda_hires -lm)
    target_link_libraries(test_cufft cuda_hires -lm)
    target_link_libraries(test_mkfile cuda_hires -lm)


    # install
    install(FILES ${CMAKE_SOURCE_DIR}/cuda/cuda_hires.h DESTINATION include)
    install(TARGETS cuda_hires LIBRARY DESTINATION lib)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/mwac_hires DESTINATION bin)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/test_cufft DESTINATION bin)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/test_mkfile DESTINATION bin)

endif()
