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
#include "adjoint_kernel.cuh"

// the usual headers
#include <stdlib.h>
#include <math.h>

// CUDA headers
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>


void init_cunfft(plan *p, dTyp *f, dTyp *x, unsigned int Ndata, unsigned int Ngrid){

	p->Ndata = Ndata;
	p->Ngrid = Ngrid;
	p->x_data = (dTyp *) malloc( Ndata * sizeof(dTyp));
	p->f_data = (dTyp *) malloc( Ngrid * sizeof(dTyp));

	
}

// computes the adjoint NFFT and stores this in plan->f_hat
void cuda_nfft_adjoint(plan *p){

	unsigned int nblocks;
	nblocks = p->Ndata / BLOCK_SIZE;
	while(nblocks * BLOCK_SIZE < p->Ndata) nblocks++;

	// move CPU data to GPU memory
	copy_data_to_gpu(p);

	// copy filter information + perform 
	// precomputation
	set_filter_properties(p);
 
	// unequally spaced data -> equally spaced grid
	fast_gridding<<<nblocks, BLOCK_SIZE>>>(p->g_f_data, 
		p->g_f_hat, p->g_x_data, p->Ngrid, p->Ndata, p->fprops);

	// (same as above, but for the filter)
	fast_gridding<<<nblocks, BLOCK_SIZE>>>(NULL, 
		p->g_f_filter, p->g_x_data, p->Ngrid, p->Ndata, p->fprops);

	// make plan
	cufftHandle cuplan;
	checkCudaErrors(cufftMakePlan1d(&cuplan, p->Ngrid, CUFFT_C2C, 1));


	// FFT(gridded data)
	checkCudaErrors(cufftExecC2C(cuplan, (cufftComplex *)(p->g_f_hat), 
							(cufftComplex *)(p->g_f_hat), CUFFT_FORWARD ));

	// FFT(filter)
	checkCudaErrors(cufftExecC2C(cuplan, (cufftComplex *)(p->g_f_filter), 
							(cufftComplex *)(p->g_f_filter), CUFFT_FORWARD ));


	// FFT(gridded data) / FFT(filter)
	nblocks = p->Ngrid / BLOCK_SIZE;
	while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++;
	divide_by_spectral_window <<< nblocks, BLOCK_SIZE >>> (p->g_f_hat, p->g_f_filter, p->Ngrid);

	// normalize (eq. 11 in Greengard & Lee 2004)	
	normalize <<< nblocks, BLOCK_SIZE >>> (p->g_f_hat, p->Ngrid);

	// Transfer back to device!
	checkCudaErrors(cudaMemcpy(p->f_hat, p->g_f_hat, p->Ngrid * sizeof(Complex),
		cudaMemcpyDeviceToHost ));

	// Free plan memory.
	checkCudaErrors(cufftDestroy(cuplan));
}

