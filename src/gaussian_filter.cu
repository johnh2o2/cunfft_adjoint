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
set_gpu_filter_properties( filter_properties *f, const dTyp *x, const int Ngrid, 
				const int Ndata );



__host__
void
generate_pinned_filter_properties(const dTyp *x, const int n, const int ng, 
		filter_properties *h_fprops, filter_properties *d_fprops, 
		cudaStream_t stream) {
	
	// set up number of blocks
	int nblocks = (n + FILTER_RADIUS) / BLOCK_SIZE;
	while (nblocks * BLOCK_SIZE < n + FILTER_RADIUS) nblocks++;

	// set filter values 
	dTyp R                  = ((dTyp) ng) / n;
        h_fprops->b             = 2 * R / (2 * R - 1) * (FILTER_RADIUS/PI);
        h_fprops->filter_radius = FILTER_RADIUS;
        h_fprops->normfac       = sqrt(2 * PI);

	// copy host filter_properties to device	
	checkCudaErrors(cudaMemcpyAsync(d_fprops, h_fprops, sizeof(filter_properties),
			cudaMemcpyHostToDevice, stream));

        // set gpu filter properties (asynchronously) [ E1, E2, E3 arrays ]
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE, 0, stream>>>(d_fprops, x, ng, n);
	checkCudaErrors(cudaGetLastError());
}
/*
__host__
void
generate_pinned_filter_properties_batch(const dTyp *x, const int *n, const int *ng, const int nlc, 
		filter_properties **h_fprops, filter_properties **d_fprops, 
		cudaStream_t stream) {
	
	// set up number of blocks
        int tot_obs = 0;
        for( int i = 0; i < nlc; i++) tot_obs += n[i];

	int nblocks = (tot_obs + nlc * FILTER_RADIUS) / BLOCK_SIZE;
	while (nblocks * BLOCK_SIZE < tot_obs + nlc * FILTER_RADIUS) nblocks++;
	
        dTyp R;
	for( int i = 0; i < nlc; i++ ) {
	// set filter values 
		R                          = ((dTyp) ng[i]) / n[i];
	        h_fprops[i]->b             = 2 * R / (2 * R - 1) * (FILTER_RADIUS/PI);
	        h_fprops[i]->filter_radius = FILTER_RADIUS;
        	h_fprops[i]->normfac       = sqrt(2 * PI);
	}

	// copy host filter_properties to device	
	checkCudaErrors(cudaMemcpyAsync(d_fprops, h_fprops, nlc * sizeof(filter_properties),
			cudaMemcpyHostToDevice, stream));

        // set gpu filter properties (asynchronously) [ E1, E2, E3 arrays ]
	set_gpu_filter_properties_batch<<<nblocks, BLOCK_SIZE, 0, stream>>>
			( d_fprops, x, ng, n, nlc);
	checkCudaErrors(cudaGetLastError());
}
*/
///////////////////////////////////////////////////////////////////////////////
// SET FILTER PROPERTIES + PRECOMPUTATIONS
__host__
void
generate_filter_properties(const dTyp *x, int n, int ng, filter_properties **fprops_host, 
		filter_properties **fprops_device) {
	// Note: need two copies of the filter properties
	//       so that we can free the (E1, E2, E3) pointers
	//       because you can't do ([GPU]pointer)->(something)
	//       
	// [CPU]fprops_host   { [GPU]E1, E2, E3, [CPU]b, normfac, filter_radius }
	// [GPU]fprops_device { [GPU] ^,  ^,  ^, [GPU]b, normfac, filter_radius }  
	//    
	dTyp b, R;

	// nthreads = nblocks x BLOCK_SIZE
	int nblocks = (n + FILTER_RADIUS) / BLOCK_SIZE;

	// make sure nthreads >= data size + filter radius
	while (nblocks * BLOCK_SIZE < n + FILTER_RADIUS) nblocks++;

	// allocate host filter_properties
	*fprops_host = (filter_properties *)malloc(sizeof(filter_properties));

	// R                :  is the oversampling factor
	R = ((dTyp) ng) / n;

	// b                :  is the characteristic length scale for the filter 
	//                     (not to be confused with the filter_radius)
        b = 2 * R / (2 * R - 1) * (FILTER_RADIUS / PI);

	// set filter radius and shape parameter of (CPU) filter_properties
	(*fprops_host)->b             = b;
	(*fprops_host)->normfac       = sqrt(2 * PI); 
	(*fprops_host)->filter_radius = FILTER_RADIUS;

	
	// allocate (GPU) filter properties
	checkCudaErrors(
		cudaMalloc((void **) fprops_device, sizeof(filter_properties))		
	);

	// allocate GPU memory for E1, E2, E3 of CPU filter_properties
	checkCudaErrors(
		cudaMalloc((void **) &((*fprops_host)->E1), n * sizeof(dTyp))		
	);
	checkCudaErrors(
		cudaMalloc((void **) &((*fprops_host)->E2), n * sizeof(dTyp))		
	);
	checkCudaErrors(
		cudaMalloc((void **) &((*fprops_host)->E3), FILTER_RADIUS * sizeof(dTyp))		
	);

	// Copy filter properties to device
	checkCudaErrors(
		cudaMemcpy(*fprops_device, *fprops_host, sizeof(filter_properties), 
						cudaMemcpyHostToDevice )
	);

	// Precompute E1, E2, E3 on GPU
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>>(*fprops_device, x, ng, n);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

///////////////////////////////////////////////////////////////////////////////
// SET UP FILTER FOR PLAN
__host__
void 
set_filter_properties(plan *p){
	LOG("in set_filter_properties");	

	// set plan filter radius
	p->filter_radius = FILTER_RADIUS;
	generate_filter_properties(p->g_x_data, p->Ndata, p->Ngrid, 
								&(p->fprops_host), &(p->fprops_device));
}


///////////////////////////////////////////////////////////////////////////////
// FREE GPU/CPU FILTER_PROPERTIES
__host__ void
free_filter_properties(filter_properties *d_fp, filter_properties *fp) {
	checkCudaErrors(cudaFree(fp->E1));
	checkCudaErrors(cudaFree(fp->E2));
	checkCudaErrors(cudaFree(fp->E3));

	checkCudaErrors(cudaFree(d_fp));
	free(fp);
}


#ifdef DOUBLE_PRECISION
#define cuExp exp
#define cuPow pow
#define cuSqrt sqrt
#else
#define cuExp expf
#define cuPow powf
#define cuSqrt sqrtf
#endif 
///////////////////////////////////////////////////////////////////////////////
// Precomputation for filter (done on GPU)
__global__
void
set_gpu_filter_properties( filter_properties *f, const dTyp *x, const int Ngrid, 
				const int Ndata ){
	// index
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( i < Ndata){
		
		// m = index of closest grid point to this data point
		int u = (int) (Ngrid * x[i] - f->filter_radius);
		
		// eps is the [0, 2pi] coordinate of the nearest gridpoint
		dTyp eps = Ngrid * x[i] - u;
		dTyp eps2 = eps / f->b;

		f->E1[i] = cuExp(- eps * eps2 ) / cuSqrt(PI * f->b);
		f->E2[i] = cuExp(    2 * eps2 ); 
	}
	else if ( i < Ndata + f->filter_radius){
		// E3 has just FILTER_RADIUS values
		int m = i - Ndata;
		f->E3[m] = cuExp( - m * m / f->b );
	}
	
}

/*
__global__
void
set_gpu_filter_properties_batch( filter_properties **fprops, const dTyp *x, const int *ng, const int *n, const int nlc) {

	// index
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        int lc = 0, s = 0;
        while (lc < nlc && i > s + n[lc]) { lc++; s += n[lc] }
        int d = i - s;

	if ( lc < nlc && d < n[lc] ){
		int ngrid = ng[lc];
		filter_properties *f = fprops[lc];

		// m = index of closest grid point to this data point
		int u = (int) (ngrid * x[i] - f->filter_radius);
		
		// eps is the [0, 2pi] coordinate of the nearest gridpoint
		dTyp eps  = ngrid * x[i] - u;
		dTyp eps2 = eps / f->b;

		f->E1[i] = cuExp(- eps * eps2 ) / cuSqrt(PI * f->b);
		f->E2[i] = cuExp(    2 * eps2 ); 
	}
	else if ( lc < nlc && d < n[lc] + fprops[lc]->filter_radius){
		// E3 has just FILTER_RADIUS values
		int m = d - n[lc];
		fprops[lc]->E3[m] = cuExp( - m * m / fprops[lc]->b );
	}
}	
*/
///////////////////////////////////////////////////////////////////////////////
// Computes filter value for a given data index, grid index, and offset (m)
__device__ 
dTyp
filter( const int j_data, const int i_grid, 
				const int m , const filter_properties *f){
	
	int mp;
	if (m < 0) mp = -m;
	else mp = m; 
	return f->E1[j_data] * cuPow(f->E2[j_data], m) * f->E3[mp];
}

///////////////////////////////////////////////////////////////////////////////
// Deconvolves filter from final result (analytically)

__global__
void
normalize(Complex *f_hat, const int Ngrid, const filter_properties *f){

	int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( k < Ngrid ){
		dTyp K = ((dTyp) k) / ((dTyp) Ngrid);
		dTyp fac = f->normfac * cuExp( K * K * f->b * 0.25 );
		f_hat[k].x *= fac;
		f_hat[k].y *= fac;
	}
}

__global__
void
normalize_bootstrap(Complex *f_hat, const int Ngrid, const int Nbootstrap, const filter_properties *f) {
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( i < Ngrid * Nbootstrap ){
		int k = i % Ngrid;
		dTyp K = ((dTyp) k) / ((dTyp) Ngrid);
		dTyp fac = f->normfac * cuExp( K * K * f->b * 0.25 );
		f_hat[i].x *= fac;
		f_hat[i].y *= fac;
	}
}
/*
__global__
void
normalize_batch(Complex *f_hat, const int *ng, const int nlc, const int idist, 
			const filter_properties **fprops) {

	int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int lc = k / idist;
	int i = k % idist;

	if ( lc < nlc && i < ng[lc] ){
		dTyp K = ((dTyp) i) / ((dTyp) ng[lc]);
		dTyp fac = fprops[lc]->normfac * cuExp( K * K * f->b * 0.25 );
		f_hat[k].x *= fac;
		f_hat[k].y *= fac;
	}
}
*/
