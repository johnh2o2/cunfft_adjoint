#ifndef UTILS_
#define UTILS_

#include "typedefs.h"

// Copies over a float array to Complex array
// TODO: Find a more efficient/sensible way to do this.
void copy_float_to_complex(dTyp *a, Complex *b, int N);

// Rescale X to [0, 2pi)
void scale_x(dTyp *x, int size);

// GPU index from ID
//__device__ unsigned int get_index();

__host__
void
init_plan(
	plan 			*p, 
	dTyp 			*f, 
	dTyp 			*x, 
	int 	Ndata, 
	int 	Ngrid
);

__host__
void 
free_plan(
	plan            *p
);

void print_plan(plan *p);

void
printComplex_d(Complex *a, int N, FILE *out);

__global__
void
printReal_d(dTyp *a, int N);

__host__
void
printReal(dTyp *a, int N);

__host__
void
printComplex(Complex *a, int N);

__global__ void print_filter_props_d(filter_properties *f, int Ndata);

#ifdef DOUBLE_PRECISION
__device__ double atomicAdd(double* address, double val);
#endif


#endif
