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

// uses a filter to map unevenly spaced data onto an evenly spaced grid
__global__ void fast_gridding( const dTyp *f_data, Complex *f_hat, const dTyp *x_data, 
                                const int Ngrid, const int Ndata, 
                                const filter_properties *fprops)
{
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	if (i < Ndata) {	
		int j = (int) ((x_data[i] + 0.5) * (Ngrid - 1));
		dTyp val;
		if (f_data == NULL)
			val = 1.0;
		else
			val = f_data[i];
		int mstart = -fprops->filter_radius + 1;
		int mend = fprops->filter_radius;
		if (j + mstart < 0) 
			mstart = -j;
		if (j + mend > Ngrid)
			mend = Ngrid - j;
		
		for (int m = mstart; m < mend; m++) 
			atomicAdd(&(f_hat[j + m].x), val * filter(i, j, m, fprops));
		
	}
}
/*
// uses a filter to map unevenly spaced data onto an evenly spaced (PADDED) grid
// { g1[0] g1[1] 0 0 0 | g2[0] g2[1] g2[2] g2[3] g2[4] | ... }
__global__ void fast_gridding_batched( const dTyp *f_data, Complex *f_hat, const dTyp *x_data, 
                        const int *Ngrid, const int *Ndata, const int nlc, const int idist, 
                        const filter_properties **fprops)
{
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int lc = 0, s = 0;
	while (i > s + Ndata[lc] && lc < nlc) {
		s+=Ndata[lc];
		lc++;
	}
	
	if (lc < nlc) {	
		int ng = Ngrid[lc];

		int j = (int) ((x_data[i] + 0.5) * (ng - 1));
		dTyp val;
		if (f_data == NULL)
			val = 1.0;
		else
			val = f_data[i];
		int mstart = -fprops->filter_radius + 1;
		int mend = fprops->filter_radius;
		if (j + mstart < 0) 
			mstart = -j;
		if (j + mend > ng)
			mend = ng - j;
		
		for (int m = mstart; m < mend; m++) 
			atomicAdd(&(f_hat[j + m + lc * idist].x), val * filter(i - s, j, m, fprops[lc]));
		
	}
}
*/

