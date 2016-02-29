/* Implements the Gaussian filter for the NFFT
 *
 * (c) John Hoffman 2016
 * jah5@princeton.edu
 * 
 */

#include "filter.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>

#define FILTER_RADIUS 6

__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const unsigned int Ngrid, 
				const unsigned int Ndata );

// pre-computes values for the filter
__host__
void 
set_filter_properties(plan *p){
	LOG("in set_filter_properties");
	// nblocks x BLOCK_SIZE threads
	unsigned int nblocks = p->Ngrid / BLOCK_SIZE;

	// Ensures that we have enough threads!
	while (nblocks * BLOCK_SIZE < p->Ngrid) nblocks++;

	p->filter_radius = FILTER_RADIUS;

	LOG("malloc p->fprops_host");
	p->fprops_host = (filter_properties *)malloc(sizeof(filter_properties));

	// R                :  is the oversampling factor
	dTyp R = ((dTyp) p->Ngrid) / p->Ndata;

	// tau              :  is the characteristic length scale for the filter 
	//                     (not to be confused with the filter_radius)
	dTyp tau = (1.0 / (p->Ngrid * p->Ngrid)) * (PI / (R* (R - 0.5))) * p->filter_radius;

	LOG("setting fprops_host->(filter_radius, tau)");
	p->fprops_host->tau = tau;
	p->fprops_host->filter_radius = p->filter_radius;

	LOG("cuda malloc p->fprops_device");
	checkCudaErrors(cudaMalloc((void **) &(p->fprops_device), sizeof(filter_properties)));

	LOG("cudaMalloc p->fprops_host->E(1,2,3)");
	checkCudaErrors(cudaMalloc((void **) &(p->fprops_host->E1), p->Ndata * sizeof(dTyp)));
	checkCudaErrors(cudaMalloc((void **) &(p->fprops_host->E2), p->Ndata * sizeof(dTyp)));
	checkCudaErrors(cudaMalloc((void **) &(p->fprops_host->E3), p->filter_radius * sizeof(dTyp)));

	checkCudaErrors(cudaDeviceSynchronize());


	LOG("cudaMemcpy p->fprops_host ==> p->fprops_device");
	checkCudaErrors(cudaMemcpy(p->fprops_device, p->fprops_host, 
		sizeof(filter_properties), cudaMemcpyHostToDevice ));

	checkCudaErrors(cudaDeviceSynchronize());

	LOG("calling setting_gpu_filter_properties");
	// Precompute E1, E2, E3 on GPU
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>>(p->fprops_device, p->g_x_data, p->Ngrid, p->Ndata);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	

	LOG("setting plan->fprops to this filter_properties pointer");
	// Set plan's filter_properties pointer to this particular filter properties object

	checkCudaErrors(cudaGetLastError());

}



/////////////////////////////////////////////
//  Uses GPU to precompute relevant values //
/////////////////////////////////////////////
__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const unsigned int Ngrid, 
				const unsigned int Ndata ){
	unsigned int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( i < Ndata){
		unsigned int m = i * Ngrid / Ndata;
		dTyp eps = x[i] - 2 * PI * m / Ngrid;
		f->E1[i] = expf(- eps * eps / (4 * f->tau));
		f->E2[i] = expf( eps * PI / (Ngrid * f->tau)); 
	}
	if ( i < f->filter_radius){
		dTyp a = PI * PI * i * i / (Ngrid * Ngrid);
		f->E3[i] = expf( -a / f->tau);

	}
	
}

__device__
dTyp
filter( const unsigned int j_data, const unsigned int i_grid, 
				const int m , filter_properties *f){
	
	unsigned int mp;
	if (m < 0) mp = -m;
	else mp = m; 
	return f->E1[j_data] * powf(f->E2[j_data], m) * f->E3[mp];
}

__global__
void
normalize(Complex *f_hat, unsigned int Ngrid, filter_properties *f){

	unsigned int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int k;
	if ( i < Ngrid ){
		k = i - Ngrid/2;
		f_hat[i].x *= sqrtf(PI/f->tau) * expf(k * k * f->tau);
	}
}
