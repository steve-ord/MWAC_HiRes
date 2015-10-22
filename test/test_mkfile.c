#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "cuda_hires.h"
void usage() {
    fprintf(stderr,"test_mkfile [-cioht] - a test wrapper for the make VCS file routines\n");
    fprintf(stderr,"\nOptions\n"); 
    fprintf(stderr,"-c\t number of channels\n"); 
    fprintf(stderr,"-i\t number of inputs\n"); 
    fprintf(stderr,"-t\t number of time steps\n"); 
    fprintf(stderr,"-o\t output file name\n");
}

int main(int argc, char **argv) {

    int ninputs = 256;
    int nchan = 128;
    int ntime = 10000;
    char *oname = strdup("test.out");

    if (argc > 1 ) {
        int c = 0;
        while ((c = getopt(argc,argv,"hi:c:o:t:")) != -1 ) {
            switch(c) {
                case 'h':
                    usage();
                    exit(-1);
                    break;
                case 'i':
                    ninputs = atoi(optarg);
                    break;
                case 'c':
                    nchan = atoi(optarg);
                    break;
                case 't':
                    ntime = atoi(optarg);
                    break;
                case 'o':
                    oname = strdup(optarg);
                    break;
                default:
                    usage();
                    break;        
            }
        }
    }
    else {
        usage();
        exit(-1);
    }


    // iname, ninputs, nchan, ntime    

    // build a set of random numbers for signal that have the same seed
    //

    // build a set of random numbers for noise that have a different seed
    //
    //
    make_vcs_file(oname,ninputs,nchan,ntime);
  //  read_vcs_file(oname,ninputs,nchan,ntime);


}   

