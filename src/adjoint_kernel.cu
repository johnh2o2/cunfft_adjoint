
#include "adjoint_kernel.h"
#include "filter.h"
#include "utils.h"
#include "typedefs.h"

__device__ 
void 
smooth_to_grid( 

	Complex 		*f_data, 
	Complex 		*f_grid, 			
	const unsigned int 	j, 
	const unsigned int 	i, 
	filter_properties 	*fprops
){

	Complex val;
	for (unsigned int m = -fprops->filter_radius + 1; 
			  m < fprops->filter_radius; 
                          m++)
	{
		if (f_data == NULL) val.x = 1.0;
		else val.x = f_data[j].x;

		val.x *= filter(j, i, m, fprops);
		atomicAdd(&(f_grid[i].x), val.x);
	}
}

__global__ 
void 
fast_gridding( 

	Complex 		*f_data, 
	Complex 		*f_grid, 
	const float 		*x_data, 
	const unsigned int 	Ngrid, 
	const unsigned int 	Ndata, 
	filter_properties 	*fprops
){
	
	unsigned int i = get_index();
	
	if (i < Ndata) {
		unsigned int j = (int) ((x_data[i] / (2 * PI)) * Ndata);
		smooth_to_grid(f_data, f_grid, j, i, fprops);
	}
}

__global__ 
void
divide_by_spectral_window( 

	Complex 		*sig, 
	const Complex 		*filt,
	const unsigned int 	N
){
	unsigned int i = get_index();
	if (i < N) sig[i].x = sig[i].x/filt[i].x;
}
