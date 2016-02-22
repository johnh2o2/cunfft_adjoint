/* Implements the adjoint NFFT
 * 
 * (c) John Hoffman 2016
 * jah5@princeton.edu
 * 
 */

// local headers
#include "typedefs.h"
#include "filter.h"
#include "utils.h"
#include "adjoint_kernel.h"

// the usual headers
#include <stdlib.h>
#include <math.h>

// CUDA headers
#include <cuda_runtime.h>
#include <cufft.h>
//#include <helper_functions.h>
#include <helper_cuda.h>



// computes the adjoint NFFT and stores this in plan->f_hat
__host__
void 
cuda_nfft_adjoint(
	plan 			*p

){

	unsigned int nblocks;
	nblocks = p->Ndata / BLOCK_SIZE;
	while(nblocks * BLOCK_SIZE < p->Ndata) nblocks++;
 
	// unequally spaced data -> equally spaced grid
	fast_gridding<<< nblocks, BLOCK_SIZE >>>(p->g_f_data, 
		p->g_f_hat, p->g_x_data, p->Ngrid, p->Ndata, p->fprops);

	// (same as above, but for the filter)
	fast_gridding<<< nblocks, BLOCK_SIZE >>>(NULL, 
		p->g_f_filter, p->g_x_data, p->Ngrid, p->Ndata, p->fprops);

	// make plan
	cufftHandle cuplan;
	checkCudaErrors(
		cufftPlan1d(&cuplan, p->Ngrid, CUFFT_C2C, 1)
	);


	// FFT(gridded data)
	checkCudaErrors(
		cufftExecC2C(cuplan, (cufftComplex *)(p->g_f_hat), 
							(cufftComplex *)(p->g_f_hat), CUFFT_FORWARD )
	);

	// FFT(filter)
	checkCudaErrors(
		cufftExecC2C(cuplan, (cufftComplex *)(p->g_f_filter), 
							(cufftComplex *)(p->g_f_filter), CUFFT_FORWARD )
	);


	// FFT(gridded data) / FFT(filter)
	nblocks = p->Ngrid / BLOCK_SIZE;
	while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++;
	divide_by_spectral_window <<< nblocks, BLOCK_SIZE >>> (p->g_f_hat, p->g_f_filter, p->Ngrid);

	// normalize (eq. 11 in Greengard & Lee 2004)	
	normalize<<< nblocks, BLOCK_SIZE >>>(p->g_f_hat, p->Ngrid, p->fprops);

	// Transfer back to device!
	checkCudaErrors(cudaMemcpy(p->f_hat, p->g_f_hat, p->Ngrid * sizeof(Complex),
		cudaMemcpyDeviceToHost ));

	// Free plan memory.
	checkCudaErrors(cufftDestroy(cuplan));
}

