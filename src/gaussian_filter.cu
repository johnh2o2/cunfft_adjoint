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

char message[100];

__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const int Ngrid, 
				const int Ndata );

__global__ void print_filter_props_d(filter_properties *f, int Ndata){
	printf("DEVICE FILTER_PROPERTIES\n\ttau = %.3e\n\tfilter_radius = %d\n", f->tau, f->filter_radius);
	for(int i = 0; i < Ndata; i++)
		printf("\tf->E1[%-3d] = %-10.3e\n", i, f->E1[i]);
	printf("\t---------------------\n");
	for(int i = 0; i < Ndata; i++)
		printf("\tf->E2[%-3d] = %-10.3e\n", i, f->E2[i]); 
	printf("\t---------------------\n");
	for(int i = 0; i < f->filter_radius; i++)
		printf("\tf->E3[%-3d] = %-10.3e\n", i, f->E3[i]); 
	printf("\t---------------------\n");
}


// pre-computes values for the filter
__host__
void 
set_filter_properties(plan *p){
	LOG("in set_filter_properties");
	

	p->filter_radius = FILTER_RADIUS;

	// nblocks x BLOCK_SIZE threads
	int nblocks = (p->Ndata + p->filter_radius) / BLOCK_SIZE;

	// Ensures that we have enough threads!
	while (nblocks * BLOCK_SIZE < p->Ndata + p->filter_radius) nblocks++;

	LOG("malloc p->fprops_host");
	p->fprops_host = (filter_properties *)malloc(sizeof(filter_properties));

	// R                :  is the oversampling factor
	dTyp R = ((dTyp) p->Ngrid) / p->Ndata;

	// tau              :  is the characteristic length scale for the filter 
	//                     (not to be confused with the filter_radius)
	//dTyp tau = (1.0 / (p->Ndata * p->Ndata)) * (PI / (R* (R - 0.5))) * p->filter_radius;
	dTyp tau = ((2 * R - 1)/ (2 * R)) * (PI / p->Ndata);


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

	sprintf(message, "\tR = %.3e\n\ttau = %.3e\n", R, tau);

	LOG("cudaMemcpy p->fprops_host ==> p->fprops_device");
	checkCudaErrors(cudaMemcpy(p->fprops_device, p->fprops_host, 
		sizeof(filter_properties), cudaMemcpyHostToDevice ));

	checkCudaErrors(cudaDeviceSynchronize());

	LOG("calling setting_gpu_filter_properties");
	// Precompute E1, E2, E3 on GPU
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>>(p->fprops_device, p->g_x_data, p->Ngrid, p->Ndata);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}



/////////////////////////////////////////////
//  Uses GPU to precompute relevant values //
////////////////////////////////////////////
__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const int Ngrid, 
				const int Ndata ){
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( i < Ndata){
		int m = (i * Ngrid) / Ndata;
		dTyp eps = x[i] - 2 * PI * m / Ngrid;

		f->E1[i] = exp(- eps * eps / (4 * f->tau));
		f->E2[i] = exp( eps * PI / (Ngrid * f->tau)); 
	}
	else if ( i < Ndata + f->filter_radius){
		int j = i - Ndata;
		dTyp a = PI * PI * j * j / (Ngrid * Ngrid);
		f->E3[j] = exp( -a / f->tau);
	}
	
}

__device__
dTyp
filter( const int j_data, const int i_grid, 
				const int m , filter_properties *f){
	
	int mp;
	if (m < 0) mp = -m;
	else mp = m; 
	return f->E1[j_data] * pow(f->E2[j_data], m) * f->E3[mp];
}

__global__
void
normalize(Complex *f_hat, int Ngrid, filter_properties *f){

	int k = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if ( k < Ngrid ){
		//k = i - Ngrid/2
		dTyp fac = sqrt(PI/f->tau) * exp(k * k * f->tau);
		f_hat[k].x *= fac/Ngrid;
		f_hat[k].y *= fac/Ngrid;
	}
}
