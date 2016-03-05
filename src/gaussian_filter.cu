/*   gaussian_filter.cu
 *   ==================
 *   
 *   Implements the Gaussian filter
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
#include "filter.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>


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
	// [CPU]p { [CPU]fprops_host   { [GPU]E1, E2, E3, [CPU]tau, filter_radius },
	//          [GPU]fprops_device { [GPU] ^,  ^,  ^, tau, filter_radius }  
	//       }


	LOG("in set_filter_properties");
	
	dTyp tau, R;

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
	tau = ((2 * R - 1)/ (2 * R)) * (PI / p->Ndata);


	LOG("setting p->fprops_host->(filter_radius, tau)");
	// set filter radius and tau of (CPU) filter_properties
	p->fprops_host->tau = tau;
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
	checkCudaErrors(cudaMemset(p->fprops_host->E1, 0, p->Ndata * sizeof(dTyp)));
	checkCudaErrors(cudaMemset(p->fprops_host->E2, 0, p->Ndata * sizeof(dTyp)));
	checkCudaErrors(cudaMemset(p->fprops_host->E3, 0, p->filter_radius * sizeof(dTyp)));

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
	if (!EQUALLY_SPACED) {
		
		set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>>
			(p->fprops_device, p->g_x_data, p->Ngrid, p->Ndata);
	
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	}

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
		int m = (i * Ngrid) / Ndata;
		
		// eps is the [0, 2pi] coordinate of the nearest gridpoint
		dTyp eps = x[i] - (2 * PI * m) / Ngrid;

		f->E1[i] = exp(- eps * eps / (4 * f->tau));
		f->E2[i] = exp( eps * PI / (Ngrid * f->tau)); 
	}
	else if ( i < Ndata + f->filter_radius){
		// E3 has just FILTER_RADIUS values
		int j = i - Ndata;
		dTyp a = PI * PI * j * j / (Ngrid * Ngrid);
		f->E3[j] = exp( -a / f->tau);
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
		dTyp fac = sqrt(f->tau / PI) * exp( -k * k * f->tau );
		f_hat[k].x *= fac;
		f_hat[k].y *= fac;
	}
}
