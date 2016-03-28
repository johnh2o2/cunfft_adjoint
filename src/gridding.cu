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
//#include <curand.h>
//#include <curand_kernel.h>

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
                if (j >= Ngrid) j = Ngrid - 1;
                if (j < 0 ) j = 0;

		dTyp val, fval;
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
		
		for (int m = mstart; m < mend; m++) {
                        fval = val * filter(i, j, m, fprops);
			atomicAdd(&(f_hat[j + m].x), fval);
		}
	}
}

__device__ unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ unsigned int rand_XORSHIFT(unsigned int seed) {
    // Xorshift algorithm from George Marsaglia's paper
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return seed;
}


// uses a filter to map unevenly spaced data onto an evenly spaced grid
__global__ void fast_gridding_bootstrap( const dTyp *f_data, Complex *f_hat, const dTyp *x_data, 
                                const int Ngrid, const int Ndata, const int Nbootstraps,
                                const filter_properties *fprops, const unsigned int seed)
{
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if (i < Ndata * Nbootstraps) {
		int d = i%Ndata;
		int j = (int) ((x_data[d] + 0.5) * (Ngrid - 1));
                //if (j >= Ngrid) j = Ngrid - 1;
                //if (j < 0 ) j = 0;

		dTyp val, fval;
		if (f_data == NULL)
			val = 1.0;
		else {
			d = rand_XORSHIFT(wang_hash(seed + i)) % Ndata;
			val = f_data[d];
		}

		int mstart = -fprops->filter_radius + 1;
		int mend = fprops->filter_radius;
		if (j + mstart < 0) 
			mstart = -j;
		if (j + mend > Ngrid)
			mend = Ngrid - j;
		int offset =( i / Ndata ) * Ngrid;
		for (int m = mstart; m < mend; m++) {
                        fval = val * filter(d, j, m, fprops);
			atomicAdd(&(f_hat[j + m + offset].x), fval);
		}
	}
}

__global__ void fast_gridding_batch(const dTyp *f_data, Complex *f_hat, const dTyp *x_data, const int *Ngrid, const int *Ndata, const int idist, const int nlc, const filter_properties **fprops) {
	
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int lc, sum=0;
        while (lc < nlc && i - sum > Ndata[lc]) {
		sum += Ndata[lc];
		lc ++;
	}
        int d = i - sum;

	if (lc < nlc) {
		int ng = Ngrid[lc];
		int j = (int) ((x_data[i] + 0.5) * (ng - 1));
		dTyp val, fval;
		if( f_data == NULL)
			val = 1.0;
		else 
			val = f_data[i];
		
		int mstart = -fprops->filter_radius + 1;
		int mend = fprops->filter_radius;
		if ( j + mstart < 0 )
			mstart = -j;
		if ( j + mend > ng )
			mend = ng - j;
		int offset = idist * lc;
		filter_properties *fp = fprops[lc];
		for (int m = mstart; m < mend; m++) {
			fval = val * filter(d, j, m, fp);
			atomicAdd(&(f_hat[j + m + offset].x), fval);
	        }
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

