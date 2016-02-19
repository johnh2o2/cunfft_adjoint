
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "kernel.cu"

void cuda_nfft_adjoint(plan *p){
	int nblocks;
	nblocks = p->Ndata / BLOCK_SIZE;
	while(nblocks * BLOCK_SIZE < p->Ndata) nblocks++;

	// move CPU data to GPU memory
	copy_data_to_gpu(p);
 
	// unequally spaced data -> equally spaced grid
	fast_gaussian_gridding<<<nblocks, BLOCK_SIZE>>>(p->g_f_data, 
		p->g_f_hat, p->g_x_data, p->Ngrid, p->Ndata);

	// (same as above, but for the filter)
	fast_gaussian_gridding<<<nblocks, BLOCK_SIZE>>>(NULL, 
		p->g_f_filter, p->g_x_data, p->Ngrid, p->Ndata);

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

	divide_by_spectral_window<<<nblocks, BLOCK_SIZE>>>(p->g_f_hat, p->g_f_filter, p->Ngrid);

	// Transfer back to device!
	checkCudaErrors(cudaMemcpy(p->f_hat, p->g_f_hat, p->Ngrid * sizeof(Complex),
		cudaMemcpyDeviceToHost ));

	// Free plan memory.
	checkCudaErrors(cufftDestroy(cuplan));
}

void
main(int argc, char *argv[]){
	
}