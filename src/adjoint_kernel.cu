/*   adjoint_kernel.cu
 *   =================   
 *   
 *   Contains filter-independent GPU operations for cuNFFT_adjoint 
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
#include "typedefs.h"
#include "adjoint_kernel.h"
#include "filter.h"
#include "utils.h"


__device__ dTyp complexModulus2(Complex a){
	return a.x * a.x + a.y * a.y;
}

__device__ Complex complexDivide(Complex a, Complex b){
	Complex solution;
	dTyp modb2 = complexModulus2(b);
	if (modb2 == 0){
		solution.x = 0;
		solution.y = 0;
		return solution;
	}

	solution.x = a.x * b.x + a.y * b.y;
	solution.y = a.y * b.x - a.x * b.y;
	solution.x /= modb2;
	solution.y /= modb2;

	return solution;

}
__device__ 
void 
smooth_to_grid( 

	Complex 		*f_data, 
	Complex 		*f_grid, 			
	const int 	i_data, 
	const int 	i_grid, 
	filter_properties 	*fprops,
	const int  Ngrid
){

	Complex val, val2;
	dTyp fval;

	if (f_data == NULL) {
		val.x = 1.0;
		val.y = 0.0;
	}
	else {
		val.x = f_data[i_data].x;
		val.y = f_data[i_data].y;
	}

	int mstart = -fprops->filter_radius + 1;
	int mend = fprops->filter_radius;

	if (i_grid + mstart < 0)
		mstart = -i_grid;
	if (i_grid + mend > Ngrid)
		mend = Ngrid - i_grid;

	for (int m = mstart; m < mend;  m++) {

		fval = filter(i_data, i_grid, m, fprops);
		
		val2.x = val.x * fval;
		val2.y = val.y * fval;

		atomicAdd(&(f_grid[i_grid + m].x), val2.x);
		atomicAdd(&(f_grid[i_grid + m].y), val2.y);
	}
}

__global__ 
void 
fast_gridding( 

	Complex 		*f_data, 
	Complex 		*f_grid, 
	const dTyp 		*x_data, 
	const int 	Ngrid, 
	const int 	Ndata, 
	filter_properties 	*fprops
){
	
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	if (i < Ndata) {
		
		int j = (int) ((x_data[i] / (2 * PI)) * Ngrid);
		smooth_to_grid(f_data, f_grid, i, j, fprops, Ngrid);
	}
}

__global__ 
void
divide_by_spectral_window( 

	Complex 		    *sig, 
	const Complex 		*filt,
	const int 	N
){
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if (i < N) 
		sig[i] = complexDivide(sig[i], filt[i]);
	
}
