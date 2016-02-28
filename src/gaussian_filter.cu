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


__global__
void
set_gpu_filter_variables( filter_properties *f, dTyp tau, dTyp filter_radius){
	f->tau = tau;
	f->filter_radius = filter_radius;
}

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

	dTyp *d_E1, *d_E2, *d_E3;
	//[p->Ndata], d_E2[p->Ndata], d_E3[p->filter_radius];
	filter_properties *f;

	LOG("cudaMalloc f");
	checkCudaErrors(cudaMalloc((void **) &f, sizeof(filter_properties)));

	LOG("cudaMalloc d_E(1,2,3)");
	checkCudaErrors(cudaMalloc((void **) &d_E1, p->Ndata * sizeof(dTyp)));
	checkCudaErrors(cudaMalloc((void **) &d_E2, p->Ndata * sizeof(dTyp)));
	checkCudaErrors(cudaMalloc((void **) &d_E3, p->filter_radius * sizeof(dTyp)));

	checkCudaErrors(cudaDeviceSynchronize());

	LOG("cudaMemcpy d_E. -> f->E.");
	checkCudaErrors(cudaMemcpy((void **) &(f->E1), &d_E1, 
		sizeof(filter_properties *), cudaMemcpyHostToDevice);

	checkCudaErrors(cudaMemcpy((void **) &(f->E2), &d_E2, 
		sizeof(filter_properties *), cudaMemcpyHostToDevice);

	checkCudaErrors(cudaMemcpy((void **) &(f->E3), &d_E3, 
		sizeof(filter_properties *), cudaMemcpyHostToDevice);

	checkCudaErrors(cudaDeviceSynchronize());

	// R                :  is the oversampling factor
	dTyp R = ((dTyp) p->Ngrid) / p->Ndata;

	// tau              :  is the characteristic length scale for the filter 
	//                     (not to be confused with the filter_radius)
	dTyp tau = (1.0 / (p->Ngrid * p->Ngrid)) * (PI / (R* (R - 0.5))) * p->filter_radius;

	LOG("Setting tau and filter radius");
	// set tau and filter_radius (has to be done on GPU)
	set_gpu_filter_variables<<< 1, 1 >>>(f, tau, FILTER_RADIUS);

	checkCudaErrors(cudaDeviceSynchronize());

	LOG("calling setting_gpu_filter_properties");
	// Precompute E1, E2, E3 on GPU
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>>(f, p->x_data, p->Ngrid, p->Ndata);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaDeviceSynchronize());
	

	LOG("setting plan->fprops to this filter_properties pointer");
	// Set plan's filter_properties pointer to this particular filter properties object
	p->fprops = f;

	checkCudaErrors(cudaGetLastError());

}



/////////////////////////////////////////////
//  Uses GPU to precompute relevant values //
/////////////////////////////////////////////
__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const unsigned int Ngrid, 
				const unsigned int Ndata ){
	unsigned int i = get_index();
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

	unsigned int i = get_index();
	int k;
	if ( i < Ngrid ){
		k = i - Ngrid/2;
		f_hat[i].x *= sqrtf(PI/f->tau) * expf(k * k * f->tau);
	}
}
