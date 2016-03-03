#ifndef ADJOINT_KERNEL_
#define ADJOINT_KERNEL_

#include "typedefs.h"

__device__ 
void 
smooth_to_grid( 
	Complex 		    *f_data, 
	Complex 		    *f_grid, 
	const int 	j, 
	const int 	i, 
	filter_properties 	*fprops,
	const int Ngrid
);

__global__ 
void 
fast_gridding(
	Complex 		    *f_data, 
	Complex 		    *f_grid, 
	const dTyp 		*x_data, 
	const int 	Ngrid, 
	const int 	Ndata, 
	filter_properties 	*fprops
);

__global__ 
void
divide_by_spectral_window(
	Complex 		    *sig, 
	const Complex 		*filt, 
	const int 	N
);

#endif
