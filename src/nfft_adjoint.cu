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


FILE *out;

__global__ void access(const Complex *a, int index) {
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if (i == 0) Complex junk = a[index];
}
// computes the adjoint NFFT and stores this in plan->f_hat
__host__
void
cuda_nfft_adjoint(
    plan 			*p

) {

	int nblocks;
	nblocks = p->Ndata / BLOCK_SIZE;
	while (nblocks * BLOCK_SIZE < p->Ndata) nblocks++;
	
	
	LOG("about to do fast_gridding");
	// unequally spaced data -> equally spaced grid
	fast_gridding<<< nblocks, BLOCK_SIZE >>>(
		           p->g_f_data,
		           p->g_f_hat,
		           p->g_x_data,
		           p->Ngrid,
		           p->Ndata,
		           p->fprops_device
	);

	
	checkCudaErrors(cudaDeviceSynchronize());

	out = fopen("gridded_data.dat", "w");
	printComplex_d(p->g_f_hat, p->Ngrid, out);
	fclose(out);


	LOG("about to do fast_gridding (filter)");
	// (same as above, but for the filter)
	fast_gridding<<< nblocks, BLOCK_SIZE >>>(
		           NULL,
		           p->g_f_filter,
		           p->g_x_data,
		           p->Ngrid,
		           p->Ndata,
		           p->fprops_device
	);

	
	checkCudaErrors(cudaDeviceSynchronize());

	out = fopen("gridded_filter.dat", "w");
	printComplex_d(p->g_f_filter, p->Ngrid, out);
	fclose(out);
	
	

	LOG("planning cufftPlan");
	// make plan
	cufftHandle cuplan;
	cufftPlan1d(
		           &cuplan,
		           p->Ngrid * sizeof(Complex),
		           CUFFT_C2C,
		           1
	);

	checkCudaErrors(cudaDeviceSynchronize());


	
	LOG("doing FFT of gridded data.");
	// FFT(gridded data)
	cufftExecC2C(  cuplan,
		          (fftComplex *)(p->g_f_hat),
				  (fftComplex *)(p->g_f_hat),
				   CUFFT_FORWARD
	);
	
	checkCudaErrors(cudaDeviceSynchronize());

	out = fopen("FFT_raw_f_hat.dat", "w");
	printComplex_d(p->g_f_hat, p->Ngrid, out);
	fclose(out);

	/*
	LOG("doing FFT of filter.");
	// FFT(filter)
	cufftExecC2C(  cuplan,
				  (fftComplex *)(p->g_f_filter),
				  (fftComplex *)(p->g_f_filter),
				   CUFFT_FORWARD
	);

	checkCudaErrors(cudaDeviceSynchronize());

	out = fopen("FFT_raw_f_filter.dat", "w");
	printComplex_d(p->g_f_filter, p->Ngrid, out);
	fclose(out);
	
	
	nblocks = p->Ngrid / BLOCK_SIZE;
	while (nblocks * BLOCK_SIZE < p->Ngrid) nblocks++;

	
	// FFT(gridded data) / FFT(filter)
	LOG("Dividing by spectral window.");
	divide_by_spectral_window <<< nblocks, BLOCK_SIZE >>> (
	    p->g_f_hat,
	    p->g_f_filter,
	    p->Ngrid
	);

	checkCudaErrors(cudaDeviceSynchronize());

	out = fopen("FFT_after_dividing_by_filter.dat", "w");
	printComplex_d(p->g_f_hat, p->Ngrid, out);
	fclose(out);
	*/

	LOG("Normalizing");
	// normalize (eq. 11 in Greengard & Lee 2004)
	normalize <<< nblocks, BLOCK_SIZE >>>(
	    p->g_f_hat,
	    p->Ngrid,
	    p->fprops_device
	);
	
	checkCudaErrors(cudaDeviceSynchronize());
	
	
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
	checkCudaErrors(cudaDeviceSynchronize());
	

	
	LOG("cufftDestroy(cuplan)");
	// Free plan memory.
	cufftDestroy(cuplan);
	
}


