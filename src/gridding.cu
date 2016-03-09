/*   gridding.cu
 *   ===========   
 *   
 *   Contains gridding subroutines for the adjoint operation 
 * 
 *   (c) John Hoffman 2016
 * 
 *   This file is part of CUNA
 *
 *   CUNA is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CUNA is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CUNA.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuComplex.h>

#include "cuna_typedefs.h"
#include "cuna_gridding.h"
#include "cuna_filter.h"
#include "cuna_utils.h"

// smooths a given datapoint onto an evenly spaced grid
__device__ void smooth_to_grid(dTyp *f_data, dTyp *f_grid, const int i_data, 
                               const int i_grid, filter_properties *fprops,
                               const int Ngrid )
{

	dTyp val, fval;
	if (f_data == NULL)
		val = 1.0;
	else
		val = f_data[i_data];
	
	int mstart = -fprops->filter_radius + 1;
	int mend = fprops->filter_radius;

	if (i_grid + mstart < 0)
		mstart = -i_grid;
	if (i_grid + mend > Ngrid)
		mend = Ngrid - i_grid;

	for (int m = mstart; m < mend;  m++) {
		fval = val * filter(i_data, i_grid, m, fprops);
		atomicAdd(&(f_grid[i_grid + m]), fval);
	}
}

// uses a filter to map unevenly spaced data onto an evenly spaced grid
__global__ void fast_gridding( dTyp *f_data, dTyp *f_grid, dTyp *x_data, 
                                const int Ngrid, const int Ndata, 
                                filter_properties *fprops)
{
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	if (i < Ndata) {	
		int j = (int) ((x_data[i] + 0.5) * (Ngrid - 1));
		smooth_to_grid(f_data, f_grid, i, j, fprops, Ngrid);
	}
}

