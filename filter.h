#ifndef FILTER_
#define FILTER_

#include "typedefs.h"

void set_filter_properties(plan *p);


__global__ 
void 
set_gpu_filter_properties( filter_properties *f, float *x, int Ngrid, int Ndata );
	

__device__ 
float 
smoothing_filter( int j, int m, int mprime);

__global__ 
void 
set_gpu_filter_properties( filter_properties *f, float *x, int Ngrid, int Ndata );

#endif