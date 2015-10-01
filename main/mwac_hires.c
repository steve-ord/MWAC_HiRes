#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "cuda_hires.h"

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
    int mode = 0; // 0 == single file 1 == multi-file
    // channel info
    int chan_select = 0;
    int nhyper = 128; // no options at the moment
    int nfine = 128; // no options at the moment
    // input info 
    int ninput = 256; /// there are 256 inputs (128 dual pol)
    int nbit = 4; // 4 bit samples
    int verbose = 1;
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
                    mode = 0;
                    break;
                case 'm':
                    list_in = strdup(optarg);
                    mode = 1;
                    break;
                case 'n':
                    chan_select = atoi(optarg);
                    break;
                case 'o':
                    output_file = strdup(optarg);
                    break;
                default:
                    usage();
                    exit(-1);
            }

        }
    }
    else {
        usage();
        exit(-1);
    }

    if (mode == 1) {
        fprintf(stderr,"Multi-input file mode not yet supported .... byeee!\n");
    
        usage();
        exit(-1);
    }

    if (mode == 0) {
        
        if (input_file == NULL || output_file == NULL) {
            fprintf(stderr,"Please supply input and output files\n");
            usage();
            exit(-1);
        }

        FILE *input = NULL;
        FILE *output = NULL;
        extern int errno;

        input = fopen(input_file,"r");
        if (input == NULL) {
            fprintf(stderr,"Failed to open input(%s) - error == %s\n",input_file,strerror(errno));
            exit(-1);
        }
        output = fopen(output_file,"w");
        if (output == NULL) {
            fprintf(stderr,"Failed to open output(%s) - error == %s\n",output_file,strerror(errno));
            exit(-1);
        }

        // some input parameters
        int ncomplex_per_input_gulp = ninput * nfine * nhyper;
        int bytes_per_input_complex = 1; // 2x4bit
        size_t input_gulp_size = ncomplex_per_input_gulp * bytes_per_input_complex;
        if (verbose)
            fprintf(stdout,"Input gulp size = %zu\n",input_gulp_size); 
        // some output parameters
        int ncomplex_per_output_gulp = ninput * nhyper;
        int bytes_per_output_complex = 8; //float2
        size_t output_gulp_size = ncomplex_per_output_gulp * bytes_per_output_complex;
        if (verbose)
            fprintf(stdout,"Output gulp size = %zu\n",output_gulp_size); 
        // some host memory
        char *h_input = (char *) malloc (input_gulp_size);
        char *h_output = (char *) malloc (output_gulp_size);
        if (h_input == NULL || h_output == NULL) {
            fprintf(stderr,"Failed to allocate host memory: %s\n",strerror(errno));
            exit(-1);
        }
        
        size_t total_bytes_read = 0;
        size_t total_gulps_read = 0;
        size_t total_bytes_written = 0;

        while (!feof(input)) {
            size_t ngulps_read = fread((void *) h_input,input_gulp_size,1,input);
            if (ngulps_read == 1) {

                total_bytes_read = total_bytes_read + ngulps_read * input_gulp_size;
                total_gulps_read = total_gulps_read + ngulps_read;

                // process
                hires_4b((complex_sample_4b_t *) h_input, (complex_sample_4b_t *) h_output,chan_select); 
                // write to output
                size_t ngulps_written = fwrite( (void *) h_output,output_gulp_size,1,output);
                if (ngulps_written != 1) {
                    fprintf(stderr,"Failed to write full gulp to output: %s\n",strerror(errno));
                }
                else {
                    total_bytes_written = total_bytes_written + ngulps_written*output_gulp_size;
                }
            }
            else {
                if (feof(input)) {
                    fprintf(stdout,"EOF on input\n");
                    fprintf(stdout,"Read %zu gulps (%zu bytes) of %d time samples from input\n",total_gulps_read,total_bytes_read,nhyper);
                    fprintf(stdout,"Wrote %zu bytes in total \n",total_bytes_written);
                }
                else {
                    fprintf(stdout,"Failed to read a full gulp from input (rval %zu) : %s\n",ngulps_read,strerror(errno));
                }
                break;
            }
        }
        fclose(input);
        fclose(output);

        free(h_input);
        free(h_output);



    }   
}
