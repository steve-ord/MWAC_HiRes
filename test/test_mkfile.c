#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "cuda_hires.h"

int main(int argc, char **argv) {
    // iname, ninputs, nchan, ntime    
    make_vcs_file("test.out",4,8,8);
    read_vcs_file("test.out",4,8,8);

}   

