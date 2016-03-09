/*   gaussian_filter.cu
 *   ==================
 *   
 *   Implements the Gaussian filter
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
#include <stdlib.h>
#include <stdio.h>

#include "cuna_filter.h"
#include "cuna_utils.h"

#ifdef DOUBLE_PRECISION
#define FILTER_RADIUS 12
#else
#define FILTER_RADIUS 6
#endif


__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const int Ngrid, 
				const int Ndata );

///////////////////////////////////////////////////////////////////////////////
// SET FILTER PROPERTIES + PRECOMPUTATIONS
__host__
void 
set_filter_properties(plan *p){

	// Note: need two copies of the filter properties
	//       so that we can free the (E1, E2, E3) pointers
	//       because you can't do ([GPU]pointer)->(something)
	//       
	// [CPU]p { [CPU]fprops_host   { [GPU]E1, E2, E3, [CPU]b, normfac, filter_radius },
	//          [GPU]fprops_device { [GPU] ^,  ^,  ^, [GPU]b, normfac, filter_radius }  
	//       }


	LOG("in set_filter_properties");
	
	dTyp b, R;

	// set plan filter radius
	p->filter_radius = FILTER_RADIUS;

	// nthreads = nblocks x BLOCK_SIZE
	int nblocks = (p->Ndata + p->filter_radius) / BLOCK_SIZE;

	// make sure nthreads >= data size + filter radius
	while (nblocks * BLOCK_SIZE < p->Ndata + p->filter_radius) nblocks++;

	// allocate host filter_properties
	LOG("malloc p->fprops_host");
	p->fprops_host = (filter_properties *)malloc(sizeof(filter_properties));

	// R                :  is the oversampling factor
	R = ((dTyp) p->Ngrid) / p->Ndata;

	// tau              :  is the characteristic length scale for the filter 
	//                     (not to be confused with the filter_radius)
	// NOTES:
	//     below was the expression I found in Greengard & Lee 2003; I think 
	//     they must have had a typo, since this tau is much too small.
	//
	//        tau = (1.0 / (p->Ndata * p->Ndata)) 
	// 			* (PI / (R* (R - 0.5))) * p->filter_radius;
	//tau = ((2 * R - 1)/ (2 * R)) * (PI / p->Ndata);
    b = 2 * R / (2 * R - 1) * (p->filter_radius / PI);

	LOG("setting p->fprops_host->(filter_radius, tau)");
	// set filter radius and shape parameter of (CPU) filter_properties
	p->fprops_host->b = b;
	p->fprops_host->normfac = sqrt(2 * PI); // * p->Ngrid;
	p->fprops_host->filter_radius = p->filter_radius;

	
	LOG("cuda malloc p->fprops_device");
	// allocate (GPU) filter properties
	checkCudaErrors(
		cudaMalloc(
			(void **) &(p->fprops_device), 
			sizeof(filter_properties)
			)
		);

	
	LOG("cudaMalloc p->fprops_host->E(1,2,3)");
	// allocate GPU memory for E1, E2, E3 of CPU filter_properties
	checkCudaErrors(
		cudaMalloc(
			(void **) &(p->fprops_host->E1), 
			p->Ndata * sizeof(dTyp)
			)
		);
	checkCudaErrors(
		cudaMalloc(
			(void **) &(p->fprops_host->E2), 
			p->Ndata * sizeof(dTyp)
			)
		);
	checkCudaErrors(
		cudaMalloc(
			(void **) &(p->fprops_host->E3), 
			p->filter_radius * sizeof(dTyp)
			)
		);

	LOG("cudaMemcpy p->fprops_host ==> p->fprops_device");
	// Copy filter properties to device
	checkCudaErrors(
		cudaMemcpy(
			p->fprops_device, 
			p->fprops_host, 
			sizeof(filter_properties), 
			cudaMemcpyHostToDevice 
			)
		);

	LOG("calling setting_gpu_filter_properties");
	// Precompute E1, E2, E3 on GPU
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>> (p->fprops_device, 
						p->g_x_data, p->Ngrid, p->Ndata);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	

}



///////////////////////////////////////////////////////////////////////////////
// Precomputation for filter (done on GPU)
__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const int Ngrid, 
				const int Ndata ){
	// index
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( i < Ndata){
		
		// m = index of closest grid point to this data point
		int u = (int) (Ngrid * x[i] - f->filter_radius);
		
		// eps is the [0, 2pi] coordinate of the nearest gridpoint
		dTyp eps = Ngrid * x[i] - u;

		f->E1[i] = exp(- eps * eps /  f->b ) / sqrt(PI * f->b);
		f->E2[i] = exp(    2 * eps /  f->b ); 
	}
	else if ( i < Ndata + f->filter_radius){
		// E3 has just FILTER_RADIUS values
		int m = i - Ndata;
		f->E3[m] = exp( - m * m / f->b );
	}
	
}

///////////////////////////////////////////////////////////////////////////////
// Computes filter value for a given data index, grid index, and offset (m)
__device__
dTyp
filter( const int j_data, const int i_grid, 
				const int m , filter_properties *f){
	
	int mp;
	if (m < 0) mp = -m;
	else mp = m; 
	return f->E1[j_data] * pow(f->E2[j_data], m) * f->E3[mp];
}

///////////////////////////////////////////////////////////////////////////////
// Deconvolves filter from final result (analytically)

__global__
void
normalize(Complex *f_hat, int Ngrid, filter_properties *f){

	int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( k < Ngrid ){
		dTyp K = ((dTyp) k) / ((dTyp) Ngrid);// - 0.5;
		dTyp fac = f->normfac * exp( K * K * f->b / 4. );
		f_hat[k].x *= fac;
		f_hat[k].y *= fac;
	}
}
