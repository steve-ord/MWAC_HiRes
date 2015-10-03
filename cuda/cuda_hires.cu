#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <complex.h>
#include "cuda_hires.h"

static __global__ void Unpack_4b(complex_sample_4b_t *, cufftComplex *, int,int,int,int);
static __global__ void BlockReorderTranspose(cufftComplex *, complex_sample_4b_t *);
static __global__ void Synthesise(cufftComplex *, float *);

void make_vcs_file(char *outname, int ninputs, int nchan, int ntime) {

    // first allocate

    int memsize = ntime*nchan*ninputs*sizeof(cufftComplex);
    int ncomponents = ntime*nchan;

    // first synthesize a timeseries for each input
 
    cufftComplex *d_x = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_x,memsize));
 
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
        h_bpass[comp] = 1.0;
    }

    // copy to device

    checkCudaErrors(cudaMemcpy(d_bpass,h_bpass,bpass_size,cudaMemcpyHostToDevice));


    for (int inp=0;inp<ninputs;inp++){
        cufftComplex *d_x_ptr = &d_x[inp*nchan*ntime];
        Synthesise<<<nchan,ntime>>>(d_x_ptr,d_bpass);
    }
    
    checkCudaErrors(cudaFree(d_x));

}




void test_fft() {
#define NSAMP 8
    cufftComplex x[NSAMP];
    float bpass[NSAMP];

    // ramp bandpass
    for (int b = 0; b < NSAMP; b++) {
        bpass[b] = b - NSAMP/2.0;
    }
    // synthesis
    // Allocate device memory
 
    cufftComplex *d_x = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_x,NSAMP*sizeof(cufftComplex)));

    float *d_bpass = NULL;
    checkCudaErrors(cudaMalloc((void **) &d_bpass,NSAMP*sizeof(float)));

    // copy to device
    checkCudaErrors(cudaMemcpy(d_bpass,bpass,NSAMP*sizeof(float),cudaMemcpyHostToDevice));

    Synthesise<<<NSAMP,1>>>(d_x,d_bpass);

    //FFT
    cufftHandle plan; 
    checkCudaErrors(cufftPlan1d(&plan, NSAMP, CUFFT_C2C, 1)); 
    checkCudaErrors(cufftExecC2C(plan, (cufftComplex *)d_x, (cufftComplex *)d_x, CUFFT_FORWARD));

    // copy back
    checkCudaErrors(cudaMemcpy(x,d_x,NSAMP*sizeof(cufftComplex),cudaMemcpyDeviceToHost));

    // check

    for (int i=0;i<NSAMP;i++) {
        
        x[i].x = x[i].x/NSAMP;
        x[i].y = x[i].y/NSAMP;

        float abs_x = x[i].x * x[i].x + x[i].y*x[i].y;
        fprintf(stdout,"%f --- %f\n",bpass[i],sqrtf(abs_x));
    }
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
    BlockReorderTranspose<<<ninputs,nchan_out>>>(d_fft_stack,d_signal);

    getLastCudaError("Kernel execution failed [ BlockReorderTranspose ]");

    // data back to host

    checkCudaErrors(cudaMemcpy(out_data,d_signal,nsamples_out,cudaMemcpyDeviceToHost));

    // free device memory
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_fft_stack));
}
static __global__ void Synthesise(cufftComplex *x, float *bpass) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int nsamp = blockDim.x * gridDim.x;

    x[n].x = 0.0;
    x[n].y = 0.0;
    
    for (int k=0 ; k < nsamp ; k++) {
        float freq = k - nsamp/2.0;
        x[n].x =  x[n].x + bpass[k]*cosf(2*M_PI*freq*n/nsamp) ;
        x[n].y =  x[n].y + bpass[k]*sinf(2*M_PI*freq*n/nsamp) ;
    }
}



static __global__ void BlockReorderTranspose(complex_sample_t *in, complex_sample_4b_t *out){

    const int input_location = blockIdx.x * blockDim.x + threadIdx.x;
    int output_location = 0;
    if (threadIdx.x < blockDim.x/2) {
        output_location = (threadIdx.x + blockDim.x/2)* gridDim.x + blockIdx.x;
    }
    else {
        output_location = (threadIdx.x - blockDim.x/2)* gridDim.x + blockIdx.x;
    }

    cufftComplex sample = in[input_location];
    int8_t sample_x = __float2int_ru(sample.x); 
    int8_t sample_y = __float2int_ru(sample.y); 
    

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
    uint64_t out_location = (input_id*blockDim.x) + time_id;

    complex_sample_4b_t sample = in_raw[ii];
    
    // mask off the lowest 4
    
    uint8_t original = sample & 0xf;
    

    if (original >= 0x8) {
       fft_stack[out_location].y = original - 0x10;
    }
    else {
       fft_stack[out_location].y = original;
    }

    sample >>= 4;

    original = sample & 0xf;
    

    if (original >= 0x8) {
       fft_stack[out_location].x = original - 0x10;
    }
    else {
       fft_stack[out_location].x = original;
    }

}
