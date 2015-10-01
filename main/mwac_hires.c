#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


void usage() {
    fprintf(stderr,"mwac_hires - a simple utility to increase the frequency resolution of MWA VCS files\n");

    fprintf(stderr,"It generates a single file (multi HDU) containing 128 hyper-fine channels. If the input number of time samples cannot make a full 128 sample input to the FFT then that sample will be dropped.\n");
    fprintf(stderr,"\nOptions:\nSingle file mode <testing>\n");
    fprintf(stderr,"\t-i <input file> -- Individual VCS file\n");
    fprintf(stderr,"\t-o <output file> --  Individual Output file\n");
    fprintf(stderr,"\t-n <input channel select> -- Which channel to hyper-fine split\n");
    fprintf(stderr,"\nMulti-file mode\n");
    fprintf(stderr,"\t-m <list of input files>\n");
    fprintf(stderr,"\t-n <input channel select>\n");
    fprintf(stderr,"\t-o <output file> Individual output file\n");
}
int main(int argc, char **argv) {

    /* pseudo code for process */
    /* 
     *  open list of input files
     *  open output file
     *
     *  while input files to process 
     *
            open input file

            while (!eof input file) 
            
                get good block of data 

                process

                write to output file 

                if (output file full)
                    open new output file

         
      *** Done */ 

    // Step 1: Get input list 

    // options: files/names
    //
    char *input_file = NULL;
    char *list_in = NULL;
    char *output_file = NULL;
    // channel info
    int chan_select = 0;
    int n_hyper = 128; // no options at the moment
    int n_fine = 128; // no options at the moment
    
    if (argc > 1) {
        int c = 0;
        while ((c = getopt(argc,argv,"hi:m:n:o:")) != -1) {
            switch (c) {
                case 'h':
                    usage();
                    exit(-1);
                    break;
                case 'i':
                    input_file = strdup(optarg);
                    break;
                case 'm':
                    list_in = strdup(optarg);
                    break;
                case 'n':
                    chan_select = atoi(optarg);
                    break;
                case 'o':
                    output_file = strdup(optarg);
                    break;
                default:
                    usage();
                    break;
            }

        }
    }
    else {
        usage();
    }
}   

