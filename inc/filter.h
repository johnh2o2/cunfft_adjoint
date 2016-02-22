/* filter.h
*  API for the smoothing filter
* 
*  (c) John Hoffman 2016
*
*/

#ifndef FILTER_
#define FILTER_

#include "typedefs.h"


__host__
void 
set_filter_properties( 
	plan                *p 
);


__device__ 
dTyp
filter( 
	const unsigned int 	j_data, 
	const unsigned int 	i_grid, 
	const int	     	m, 
	filter_properties 	*f
);

__global__
void
normalize(
	Complex 		    *f_hat, 
	unsigned int 		Ngrid,
	filter_properties 	*f
);

#endif
