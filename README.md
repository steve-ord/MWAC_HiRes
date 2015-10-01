# MWAC_HiRes
High Frequency Resolution MWAC file converter.

Simple CUDA utility to take an MWA VCS output file of 100us x 10KHz x 256 input and increase the frequecy resolution by NFACTOR.

It operates using the cuFFT library and requires a CUDA/NVIDIA GPU to run.

1) Installation

    This uses the CMake cross-platform make to build. This works in two stages

    a) define a build location. This will only be used to build the package
    b) cd <build_dir>
    c) cmake -DCMAKE_INSTALL_PREFIX=<install_location> <path to source_dir>
    d) make
    e) make install

    
