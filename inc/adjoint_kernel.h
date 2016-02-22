#ifndef ADJOINT_KERNEL_
#define ADJOINT_KERNEL_

#include "typedefs.h"

__device__ 
void 
smooth_to_grid( 
	Complex 		*f_data, 
	Complex 		*f_grid, 
	const unsigned int 	j, 
	const unsigned int 	i, 
	filter_properties 	*fprops
);

__global__ 
void 
fast_gridding(
	Complex 		*f_data, 
	Complex 		*f_grid, 
	const float 		*x_data, 
	const unsigned int 	Ngrid, 
	const unsigned int 	Ndata, 
	filter_properties 	*fprops
);

__global__ 
void
divide_by_spectral_window(
	Complex 		*sig, 
	Complex 		*filt, 
	const unsigned int 	N
);

#endif
