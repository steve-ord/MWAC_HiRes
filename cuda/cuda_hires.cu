#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <complex.h>
#include "cuda_hires.h"

static __global__ void Unpack_4b(complex_sample_4b_t *, cufftComplex *, int,int,int,int);
static __global__ void BlockReorderTo4b(cufftComplex *, complex_sample_4b_t *, int);
static __global__ void BlockReorder(cufftComplex *, cufftComplex *,int);
static __global__ void Synthesise(cufftComplex *, float *, cufftComplex *);


void get_normal_random (float U, float V, float *ans) {

    // Box-Muller

    *ans = (sqrt(-2*logf(U))*cosf(2.0*M_PI*V));
    ans++;
    *ans = (sqrt(-2*logf(U))*sinf(2.0*M_PI*V));
}

void read_vcs_file(char *iname, int ninputs, int nchan, int ntime) {

    FILE *input=NULL;

    input = fopen(iname,"r");

    // some input parameters
    int ncomplex_per_input_gulp = ninputs * nchan * ntime;
    int bytes_per_input_complex = 1; // 2x4bit
    size_t input_gulp_size = ncomplex_per_input_gulp * bytes_per_input_complex;
 
    char *h_input = (char *) malloc (input_gulp_size);
    cufftComplex *h_unpacked = (cufftComplex *) malloc (ntime*nchan*sizeof(cufftComplex));
    bzero(h_unpacked,ntime*nchan*sizeof(cufftComplex));

    size_t ngulps_read = fread((void *) h_input,input_gulp_size,1,input);
    
    int nsamples_in = ninputs*nchan*ntime; 
    
    // Allocate device memory
    // input data is a 4 bit sample for ninputs * nchan_in * nchan_out (time_samples)
    // we need to pull out a nchan_out time_samples for channel==chan_select for each input
    // After the FFT we output a single time sample for an nchan_out FB for all the inputs
    // repacked into the correct order
    
    complex_sample_4b_t *d_signal;
    checkCudaErrors(cudaMalloc((void **) &d_signal,nsamples_in*1));
     
    complex_sample_t *d_stack;
    checkCudaErrors(cudaMalloc((void **) &d_stack,nsamples_in*sizeof(complex_sample_t)));

    // copy data to Device
    checkCudaErrors(cudaMemcpy(d_signal,h_input,nsamples_in,cudaMemcpyHostToDevice));

    // unpack the 4b to complex - also transpose the input
    // input order in [ntime][nfreq][input]
    // output order is a single channel now in 
    // [ninput][ntime]

    Unpack_4b<<<ntime,ninputs>>>(d_signal, d_stack, 4 ,nchan,1,ninputs);

    // to reiterate we now have ntime samples from a single channel for 
    // all inputs
    // copy data to Host

    checkCudaErrors(cudaMemcpy(h_unpacked,d_stack,ntime*ninputs*sizeof(cufftComplex),cudaMemcpyDeviceToHost));
    
    for (int inp=0;inp<ninputs;inp++){ 
        for (int tim=0;tim<ntime;tim++){   
            cufftComplex sample = h_unpacked[inp*ntime + tim];
            fprintf(stdout,"Input %d Sampnum %d value (x + iy) %f + %f(i)\n",inp,tim,sample.x,sample.y);
        }
    }

    getLastCudaError("Kernel execution failed [ Unpack_4b ]");


    checkCudaErrors(cudaFree(d_stack));
    checkCudaErrors(cudaFree(d_signal));

}

void make_vcs_file(char *outname, int ninputs, int nchan, int ntime) {

    // first allocate

    int memsize = nchan*ninputs*sizeof(cufftComplex);
    int ncomponents = nchan;
    extern int verbose;
    extern const double passband[128];
    extern const double profile[64];

    cufftComplex *h_x = (cufftComplex *) malloc(memsize);

    // first synthesize a timeseries for each input
 
    cufftComplex *d_x = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_x,memsize));
 
    cufftComplex *d_test = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_test,memsize));


    // some device memory for the 4b data

    complex_sample_4b_t *d_sig = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_sig,nchan*ninputs));

    int bpass_size = ncomponents*sizeof(float);

    float *d_bpass = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_bpass,bpass_size));

    float *h_bpass = NULL;
    h_bpass = (float *) malloc(bpass_size);


    // Each input has nchan*ntime samples
    // this generates an array of [ninput][nsample] timeseries
    
    // it does this by generating the frequency components (+/- freq/2)
    // for this it need a bandpass
    // there are the as many frequency components as samples

    for (int comp=0;comp<ncomponents;comp++) {
        if (ncomponents == 128) {
            h_bpass[comp] = passband[comp];
        }
        else {
            h_bpass[comp] = comp;
        }
    }

    // copy to device

    checkCudaErrors(cudaMemcpy(d_bpass,h_bpass,bpass_size,cudaMemcpyHostToDevice));

    cufftHandle plan; 
    checkCudaErrors(cufftPlan1d(&plan, nchan, CUFFT_C2C, ninputs)); 

    FILE *fout = NULL;

    fout = fopen(outname,"w");

    if (fout == NULL) {
        fprintf(stderr,"Failed to open %s\n",outname);
        exit(-1);
    }

    // we need to add noise with the same seed 
    // for the signal

    char signal_state[64];
    initstate(1,signal_state,64);
    char noise_state[64];
    initstate(1024,noise_state,64);

    size_t signal_size = nchan*sizeof(cufftComplex);
    size_t noise_size = nchan*ninputs*sizeof(cufftComplex);


    cufftComplex *h_signal_array= (cufftComplex *) malloc(signal_size);
    cufftComplex *h_noise_array= (cufftComplex *) malloc(noise_size);

    cufftComplex *d_noise_array = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_noise_array,noise_size));



    //quadrature sampling each (10kHz) channel
    //
    float fc = 5000.0;
    
    for (int tim=0;tim<ntime;tim++) {
        
        fprintf(stderr,"done %d of %d\n",tim,ntime);

        char *cur_state = setstate(signal_state);

        memcpy(noise_state,cur_state,64);
        // I need to add some structure to the time domain
        // best way to do this is with a nice pulsar profile ....
    
        double value = profile[tim%64];

        for (int ch = 0 ; ch < nchan ; ch++) {
            float U=0;
            float V=0;
            while (U == 0 || U == 1) {
                U = (random() % 1000)/1000.0;
            }
            while (V == 0 || V == 1) {
                V = (random() % 1000)/1000.0;
            }

            float ans[2];
            get_normal_random(U,V,&ans[0]);
            h_signal_array[ch].x = value*ans[0]/10.0;
            h_signal_array[ch].y = value*ans[1]/10.0;
        } 

        cur_state = setstate(noise_state);

        memcpy(signal_state,cur_state,64);

        for (int inp=0;inp<ninputs;inp++){
            for (int ch = 0 ; ch < nchan ; ch++) {
                float U=0;
                float V=0;
                while (U == 0 || U == 1) {
                    U = (random() % 1000)/1000.0;
                }
                while (V == 0 || V == 1) {
                    V = (random() % 1000)/1000.0;
                }

                float ans[2];

                get_normal_random(U,V,&ans[0]);


                // h_noise_array[inp*nchan+ch].x = sinf(2.0*M_PI*fc*float(tim)/ntime) * (5.0*ans[0] + h_signal_array[ch].x);
                // h_noise_array[inp*nchan+ch].y = cosf(2.0*M_PI*fc*float(tim)/ntime) * (5.0*ans[0] + h_signal_array[ch].x);
            
                h_noise_array[inp*nchan+ch].x = (2.0*ans[0] + h_signal_array[ch].x);
                h_noise_array[inp*nchan+ch].y = (2.0*ans[1] + h_signal_array[ch].y);
            }

        }
        
        // memcpy 

        checkCudaErrors(cudaMemcpy(d_noise_array,h_noise_array,noise_size,cudaMemcpyHostToDevice));
        
        //

        for (int inp=0;inp<ninputs;inp++){
            cufftComplex *d_x_ptr = &d_x[inp*nchan];
            cufftComplex *d_noise_array_ptr = &d_noise_array[inp*nchan];

            // get nchan samples
            // these will be the same for each time-step
            // and the same for each input
            // but each sample should be different 
            // and more noise with a different (antenna based) seed for the noise

            // We need to maintain a phase relationship that is easy to re-construct
            // between the different signals - I suggest that we just rotate the phase of the 
            // random complex number by the input number

            // perhaps initially we do not rotate the phase ....
            if (nchan == 128) {
                Synthesise<<<nchan,1>>>(d_x_ptr,d_bpass,d_noise_array_ptr);
            }
            else {
                Synthesise<<<nchan,1>>>(d_x_ptr,d_bpass,NULL);
            }
        }
    
        // FFT into channels
        // input array [ntime][inputs][nchan]
        if (nchan != 128)
            checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_x, (cufftComplex *)d_x, CUFFT_FORWARD));

        /* now we need the re-order and transpose */

        BlockReorderTo4b<<<ninputs,nchan>>>(d_x,d_sig,1);
    
        checkCudaErrors(cudaMemcpy(h_x,d_sig,ninputs*nchan,cudaMemcpyDeviceToHost));

        fout = fopen(outname,"a");

        fwrite(h_x,ninputs*nchan,1,fout);

        fclose(fout);
// For comparision
        BlockReorder<<<ninputs,nchan>>>(d_x,d_test,1);

        checkCudaErrors(cudaMemcpy(h_x,d_test,ninputs*nchan*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

        for (int i=0;i<nchan;i++) {
            for (int inp=0;inp<ninputs;inp++) {
                int ii = i*ninputs + inp; 
                h_x[ii].x = h_x[ii].x;
                h_x[ii].y = h_x[ii].y;

                float abs_x = h_x[ii].x * h_x[ii].x + h_x[ii].y*h_x[ii].y;
                float test_abs = 0;
                float multiplier = 0;
                int index = 0;
                    

                if (nchan == 128) {
                    if (i<nchan/2) {
                        index = i+nchan/2;
                    }
                    else {
                        index = i-nchan/2;
                    }
                    multiplier = h_bpass[index]*h_bpass[index];
                    test_abs = multiplier*(h_noise_array[inp*nchan + index].x * h_noise_array[inp*nchan + index].x + h_noise_array[inp*nchan + index].y*h_noise_array[inp*nchan + index].y);
                    if (verbose)
                        fprintf(stdout,"input %d: ch %d (predicted) %f (%f %f(i)) --- (actual) %f (%f+%f(i))\n",inp,i,sqrtf(test_abs),h_bpass[index]*h_noise_array[inp*nchan + index].x,h_bpass[index]*h_noise_array[inp*nchan + index].y, sqrtf(abs_x),h_x[ii].x,h_x[ii].y) ;
                }
                else {

                    test_abs = h_bpass[i]* h_bpass[i];
                    if (verbose)
                        fprintf(stdout,"input %d: ch %d (predicted) %f (%f %f(i)) --- (actual) %f (%f+%f(i))\n",inp,i,sqrtf(test_abs),h_bpass[i],h_bpass[i], sqrtf(abs_x),h_x[ii].x,h_x[ii].y) ;
                }

            }

        }

    }
//
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_bpass));

}




void test_fft() {
#define NSAMP 8
#define NINPUT 4

    cufftComplex x[NSAMP*NINPUT];
    float bpass[NSAMP];

    // ramp bandpass
    for (int b = 0; b < NSAMP; b++) {
        bpass[b] = b;
    }
    // synthesis
    // Allocate device memory
 
    cufftComplex *d_x = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_x,NSAMP*NINPUT*sizeof(cufftComplex)));
 
    cufftComplex *d_x_r = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_x_r,NSAMP*NINPUT*sizeof(cufftComplex)));


    float *d_bpass = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_bpass,NSAMP*sizeof(float)));

    // copy to device
    checkCudaErrors(cudaMemcpy(d_bpass,bpass,NSAMP*sizeof(float),cudaMemcpyHostToDevice));
    
    for (int inp = 0; inp <NINPUT;inp++) {
        cufftComplex *d_ptr = &d_x[inp*NSAMP];
        Synthesise<<<NSAMP,1>>>(d_ptr,d_bpass,NULL);
    }

    //FFT
    cufftHandle plan; 
    checkCudaErrors(cufftPlan1d(&plan, NSAMP, CUFFT_C2C, NINPUT)); 
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_x, (cufftComplex *)d_x, CUFFT_FORWARD));

    //Reorder
    BlockReorder<<<NINPUT,NSAMP>>>(d_x,d_x_r,NSAMP);

    // copy back
    checkCudaErrors(cudaMemcpy(x,d_x_r,NSAMP*NINPUT*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    // check
    for (int i=0;i<NSAMP;i++) {
        for (int inp=0;inp<NINPUT;inp++) {
            int ii = i*NINPUT + inp; 
            x[ii].x = x[ii].x;
            x[ii].y = x[ii].y;

            float abs_x = x[ii].x * x[ii].x + x[ii].y*x[ii].y;
            fprintf(stdout,"input %d: ch %d (predicted) %f --- (actual) %f (%f+i%f)\n",inp,i,bpass[i],sqrtf(abs_x),x[ii].x,x[ii].y);
        }
    }
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_bpass));
}

void hires_4b(complex_sample_4b_t *in_data, complex_sample_4b_t *out_data, int chan_select, int nchan_in, int nchan_out, int ninputs) {
    
    // setDevice
    
    // Allocate device memory
    // input data is a 4 bit sample for ninputs * nchan_in * nchan_out (time_samples)
    // we need to pull out a nchan_out time_samples for channel==chan_select for each input
    // After the FFT we output a single time sample for an nchan_out FB for all the inputs
    // repacked into the correct order
    
    // is it worth stacking the FFT?

    int nsamples_in = ninputs * nchan_in * nchan_out;
    int nsamples_out = ninputs * nchan_out;

    // size of the FFT stack
    // ninput rows of nchan_out complex samples
    int nsamples_fft_stack = ninputs*nchan_out;

    complex_sample_4b_t *d_signal;
    checkCudaErrors(cudaMalloc((void **) &d_signal,nsamples_in*1));
    
    complex_sample_t *d_fft_stack;
    checkCudaErrors(cudaMalloc((void **) &d_fft_stack,nsamples_fft_stack*sizeof(complex_sample_t)));

    // copy data to Device
    checkCudaErrors(cudaMemcpy(d_signal,in_data,nsamples_in,cudaMemcpyHostToDevice));

    // call kernel to expand input data to float
    // assumes number of time samples == number of output channels
    // also extracts the time series for this channel and all inputs
    Unpack_4b<<<nchan_out,ninputs>>>(d_signal, d_fft_stack, chan_select,nchan_in,nchan_out,ninputs);
    getLastCudaError("Kernel execution failed [ Unpack_4b ]");

    // call kernel to batch FFT 
    // this will FFT all inputs at once
    cufftHandle plan; 
    checkCudaErrors(cufftPlan1d(&plan, nchan_out, CUFFT_C2C, ninputs)); 
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_fft_stack, (cufftComplex *)d_fft_stack, CUFFT_FORWARD));

    // call kernels to repack
    // We will perform the reordering and repacking in a single CUDA Kernel
    // We have to reorder the fft_stack which is now [input][channel] (channels moving fastest) to [channel][input]
    // and we need to reorder the channeld by block shifting the second half of the array to the beginning

    // probably best to do this in 2 stages:
   
    // channel reordering
 
    // & transpose & float to 4
    BlockReorderTo4b<<<ninputs,nchan_out>>>(d_fft_stack,d_signal,1);

    getLastCudaError("Kernel execution failed [ BlockReorderTo4b ]");

    // data back to host

    checkCudaErrors(cudaMemcpy(out_data,d_signal,nsamples_out,cudaMemcpyDeviceToHost));

    // free device memory
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_fft_stack));
}
static __global__ void Synthesise(cufftComplex *x, float *bpass, cufftComplex *noise) {

    // Synthesise the n'th time sample
    // using the addition of +/- nsamp/2 frequency components

    // This is modulated by the bandpass
    // If a noise spectrum (containing signal + noise) is present

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int nsamp = blockDim.x * gridDim.x;

    x[n].x = 0.0;
    x[n].y = 0.0;
    if (noise == NULL) {
        for (int k=0 ; k < nsamp ; k++) {
            float freq = k - nsamp/2.0;
            x[n].x =  x[n].x + bpass[k]*cosf(2*M_PI*freq*n/nsamp) ;
            x[n].y =  x[n].y + bpass[k]*sinf(2*M_PI*freq*n/nsamp) ;
        }
    }
    else { 
       int in_index = 0;
       int out_index = n;  
       if (n < blockDim.x/2) { 
                in_index = n+blockDim.x/2;
                x[n].x =  bpass[in_index]*noise[in_index].x;
                x[n].y =  bpass[in_index]*noise[in_index].y;
       }
       else {
                in_index = n-blockDim.x/2;
                x[n].x =  bpass[in_index]*noise[in_index].x;
                x[n].y =  bpass[in_index]*noise[in_index].y;
       }
    }
}


static __global__ void BlockReorder(complex_sample_t *in, complex_sample_t *out,int scale){

    const int input_location = blockIdx.x * blockDim.x + threadIdx.x;
    int output_location = 0;
    if (threadIdx.x < blockDim.x/2) {
        output_location = (threadIdx.x + blockDim.x/2)* gridDim.x + blockIdx.x;
    }
    else {
        output_location = (threadIdx.x - blockDim.x/2)* gridDim.x + blockIdx.x;
    }
    out[output_location].x = in[input_location].x/scale; // normalise
    out[output_location].y = in[input_location].y/scale; // normalise
}



static __global__ void BlockReorderTo4b(complex_sample_t *in, complex_sample_4b_t *out, int scale){

    /* the blockDim is essentially the number of channels */
    /* gridDim is the number of Inputs */

    const int input_location = blockIdx.x * blockDim.x + threadIdx.x;
    int output_location = 0;
    if (threadIdx.x < blockDim.x/2) {
        output_location = (threadIdx.x + blockDim.x/2)* gridDim.x + blockIdx.x;
    }
    else {
        output_location = (threadIdx.x - blockDim.x/2)* gridDim.x + blockIdx.x;
    }

    cufftComplex sample;
    sample.x = in[input_location].x/scale; // normalise
    sample.y = in[input_location].y/scale; // normalise
    int8_t sample_x = (int8_t) (sample.x); 
    int8_t sample_y = (int8_t) (sample.y); 
    

    out[output_location] = 0x0;
    out[output_location] = (sample_y & 0xf);
    
    sample_x = sample_x & 0xf;
    out[output_location] = out[output_location] | (sample_x << 4);

}
static __global__ void Unpack_4b(complex_sample_4b_t *in_raw, complex_sample_t *fft_stack, int chan_select,int nchan_in,int nchan_out, int ninput){
    // we only need to unpack those entries corresponding to the 
    // channel we wish to process
    // the input data is in [time][freq][input] order - input moving fastest
    
    // this is called with ntime blocks of ninput threads

    int input_sample_offset = ninput*chan_select;
    const int input_id = threadIdx.x;
    const int time_id = blockIdx.x;

    uint64_t ii = (time_id*ninput*nchan_in) + input_id + input_sample_offset;

    // unpack this sample and put the result in 
    // in transposed FFT_STACK at:
    // output order is now just [ninput][ntime]

    uint64_t out_location = (input_id*gridDim.x) + time_id;

    complex_sample_4b_t sample = in_raw[ii];
    
    // mask off the lowest 4
    
    uint8_t original = sample & 0xf;

    if (original >= 0x8) {
       fft_stack[out_location].y = (float) (original - 0x10);
    }
    else {
       fft_stack[out_location].y = (float) (original);
    }

    sample >>= 4;

    original = sample & 0xf;
    

    if (original >= 0x8) {
       fft_stack[out_location].x = (float) (original - 0x10);
    }
    else {
       fft_stack[out_location].x = (float) (original);
    }

}
