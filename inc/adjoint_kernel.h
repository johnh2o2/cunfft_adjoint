/*   adjoint_kernel.h
 *   ================   
 *   
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
