/* Implements the adjoint NFFT
 * 
 * (c) John Hoffman 2016
 * jah5@princeton.edu
 * 
 */

// local headers

#include "nfft_adjoint.h"

// the usual headers
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CUDA headers
//#include <cuda_runtime.h>
#include <cufft.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>



__global__ void access(const Complex *a, unsigned int index){
	Complex junk = a[index];
}
// computes the adjoint NFFT and stores this in plan->f_hat
__host__
void 
cuda_nfft_adjoint(
	plan 			*p

){
	int i;
	for (i=0; i < p->Ngrid; p++) {
		fprintf(stderr, "p->g_f_filter: %d/%d\n", i+1, p->Ngrid);
		access <<< nblocks, BLOCK_SIZE >>> (p->g_f_filter, p->Ngrid);
		checkCudaErrors(cudaGetLastError());
	}

	unsigned int nblocks;
	nblocks = p->Ndata / BLOCK_SIZE;
	while(nblocks * BLOCK_SIZE < p->Ndata) nblocks++;
 
 	LOG("about to do fast_gridding");
	// unequally spaced data -> equally spaced grid
	fast_gridding<<< nblocks, BLOCK_SIZE >>>(
		           p->g_f_data, 
		           p->g_f_hat, 
		           p->g_x_data, 
		           p->Ngrid, 
		           p->Ndata, 
		           p->fprops
	);

	checkCudaErrors(cudaGetLastError());

	LOG("about to do fast_gridding (filter)");
	// (same as above, but for the filter)
	fast_gridding<<< nblocks, BLOCK_SIZE >>>(
		           NULL, 
		           p->g_f_filter, 
		           p->g_x_data, 
		           p->Ngrid, 
		           p->Ndata, 
		           p->fprops
	);

	checkCudaErrors(cudaGetLastError());

	LOG("planning cufftPlan");
	// make plan
	cufftHandle cuplan;
	cufftPlan1d(
		           &cuplan, 
		           p->Ngrid, 
		           CUFFT_C2C, 
		           1
    );

	checkCudaErrors(cudaGetLastError());


	LOG("doing FFT of gridded data.");
	// FFT(gridded data)
	cufftExecC2C(  cuplan, 
		          (cufftComplex *)(p->g_f_hat), 
				  (cufftComplex *)(p->g_f_hat), 
				   CUFFT_FORWARD 
	);

	checkCudaErrors(cudaGetLastError());

	LOG("doing FFT of filter.");
	// FFT(filter)
	cufftExecC2C(  cuplan, 
				  (cufftComplex *)(p->g_f_filter), 
				  (cufftComplex *)(p->g_f_filter), 
				   CUFFT_FORWARD 
	);

	checkCudaErrors(cudaGetLastError());

	// FFT(gridded data) / FFT(filter)
	nblocks = p->Ngrid / BLOCK_SIZE;
	while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++;

	checkCudaErrors(cudaGetLastError());

	for (i=0; i < p->Ngrid; p++) {
		fprintf(stderr, "p->g_f_filter: %d/%d\n", i+1, p->Ngrid);
		access <<< nblocks, BLOCK_SIZE >>> (p->g_f_filter, p->Ngrid);
		checkCudaErrors(cudaGetLastError());
	}

	LOG("Dividing by spectral window.");
	divide_by_spectral_window <<< nblocks, BLOCK_SIZE >>> (
			       p->g_f_hat, 
			       p->g_f_filter, 
			       p->Ngrid
	);

	checkCudaErrors(cudaGetLastError());

	LOG("Normalizing");
	// normalize (eq. 11 in Greengard & Lee 2004)	
	normalize<<< nblocks, BLOCK_SIZE >>>(
		           p->g_f_hat, 
		           p->Ngrid, 
		           p->fprops
    );

    checkCudaErrors(cudaGetLastError());

	LOG("Transferring data back to device");
	// Transfer back to device!
	checkCudaErrors(
		cudaMemcpy(
			       p->f_hat, 
			       p->g_f_hat, 
			       p->Ngrid * sizeof(Complex),
		           cudaMemcpyDeviceToHost 
		)
    );

	LOG("cufftDestroy(cuplan)");
	// Free plan memory.
	cufftDestroy(cuplan);
}


