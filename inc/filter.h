/*   filter.h
 *   ========   
 *   
 *   Defines API for implementing a filter
 * 
 *   (c) John Hoffman 2016
 * 
 *   This file is part of cuNFFT_adjoint
 *
 *   cuNFFT_adjoint is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   cuNFFT_adjoint is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with cuNFFT_adjoint.  If not, see <http://www.gnu.org/licenses/>.
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
	const int 	j_data, 
	const int 	i_grid, 
	const int	     	m, 
	filter_properties 	*f
);

__global__
void
normalize(
	Complex 		    *f_hat, 
	int 		Ngrid,
	filter_properties 	*f
);

#endif
