add_executable(mwac_hires mwac_hires.c)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/mwac_hires DESTINATION bin)
