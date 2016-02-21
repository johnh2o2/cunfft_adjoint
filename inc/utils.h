#ifndef UTILS_
#define UTILS_

#include "typedefs.h"

// Copies over a float array to Complex array
// TODO: Find a more efficient/sensible way to do this.
void copy_float_to_complex(float *a, Complex *b, size_t N);

// Allocates and transfers plan data to GPU memory
void copy_data_to_gpu(plan *p);

// Rescale X to [0, 2pi)
void scale_x(float *x, size_t size);

// GPU index from ID
__device__ unsigned int get_index();

// normalizes FFT (see eq. 11 in Greengard & Lee 2004)
__global__ void normalize(Complex *f_hat, unsigned int Ngrid);

#endif
