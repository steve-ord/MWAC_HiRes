#include "helper_cuda.h"
#include "helper_functions.h"
#include "cuda_hires.h"



void hires_4b(complex_sample_4b_t *in_data, complex_sample_4b_t *out_data, int chan_select, int nchan_in, int nchan_out, int ninputs) {
    
    // setDevice
    
    // Allocate device memory
    // input data is a 4 bit sample for ninputs * nchan_in * nchan_out (time_samples)
    // we need to pull out a nchan_out time_samples for channel==chan_select for each input
    // After the FFT we output a single time sample for an nchan_out FB for all the inputs
    // repacked into the correct order
    
    // is it worth stacking the FFT?

    int nsamples_in = ninputs * nchan_in * nchan_out;

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

    // call kernel to extract the time series for this channel and all inputs

    // call kernel to batch FFT 

    // call kernel to repack

    // free device memory
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_fft_stack));
}
