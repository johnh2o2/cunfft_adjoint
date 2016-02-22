#ifndef UTILS_
#define UTILS_

#include "typedefs.h"

// Copies over a float array to Complex array
// TODO: Find a more efficient/sensible way to do this.
void copy_float_to_complex(float *a, Complex *b, unsigned int N);

// Rescale X to [0, 2pi)
void scale_x(float *x, unsigned int size);

// GPU index from ID
__device__ unsigned int get_index();

__host__
void
init_plan(
	plan 			*p, 
	dTyp 			*f, 
	dTyp 			*x, 
	unsigned int 	Ndata, 
	unsigned int 	Ngrid
);

__host__
void 
free_plan(
	plan            *p
);

#endif
