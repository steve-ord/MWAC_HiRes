#ifndef __CUDA_HIRES_H
#define __CUDA_HIRES_H

#include <stdint.h>
#include <cuComplex.h>

typedef uint8_t complex_sample_4b_t;      // type for 4 bit sample pair
typedef cuFloatComplex complex_sample_t;

#ifdef __cplusplus
extern "C" {
#endif    
    void hires_4b(complex_sample_4b_t *in_data, complex_sample_4b_t *out_data, int chan_select);

#ifdef __cplusplus
}
#endif

#endif

