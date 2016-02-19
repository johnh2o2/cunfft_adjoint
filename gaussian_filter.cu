/* Implements the Gaussian filter for the NFFT
 *
 * (c) John Hoffman 2016
 * jah5@princeton.edu
 * 
 */

#include "filter.h"
#include "filter_kernel.cu"

// pre-computes values for the filter
void set_filter_properties(plan *p){

	// nblocks x BLOCK_SIZE threads
	int nblocks = p->Ngrid / BLOCK_SIZE;

	// Ensures that we have enough threads!
	while (nblocks * BLOCK_SIZE < n) nblocks++;

	// malloc memory for filter_properties (on GPU)
	filter_properties *f;
	checkCudaErrors(cudaMalloc((void **) &f, sizeof(filter_properties *) ));

	checkCudaErrors(cudaMalloc((void **) &(f->E1), p->Ndata * sizeof(float) ));
	checkCudaErrors(cudaMalloc((void **) &(f->E2), p->Ndata * sizeof(float) ));
	checkCudaErrors(cudaMalloc((void **) &(f->E3), p->filter_radius * sizeof(float) ));


	// R    :  is the oversampling factor
	float R = ((float) p->Ngrid) / p->Ndata;

	// tau  :  is the characteristic length scale for the filter 
	//         (not to be confused with the filter_radius)
	float tau = (1.0 / (p->Ngrid * p->Ngrid)) * ( PI / (R* (R - 0.5)) ) * p->filter_radius;

	// Copy over R and tau to GPU
	checkCudaErrors(cudaMemcpy(&(f->tau),&tau, sizeof(float), cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy(&(f->filter_radius),&(p->filter_radius), 
		sizeof(int), cudaMemcpyHostToDevice ));

	// Precompute E1, E2, E3 on GPU
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>>(f, p->x_data, p->Ngrid, p->Ndata);

	// set the global GPU pointer to this particular filter properties object
	g_fprops = f;

}