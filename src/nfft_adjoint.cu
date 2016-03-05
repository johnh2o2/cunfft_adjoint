/*   nfft_adjoint.cu
 *   ===============   
 *   
 *   Implementation of the adjoint NFFT operation
 * 
 *   (c) John Hoffman 2016
 * 
 *   This file is part of cuNFFT_adjoint
 *
 *   cuNFFT_adjoint is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   cuNFFT_adjoint is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with cuNFFT_adjoint.  If not, see <http://www.gnu.org/licenses/>.
 */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CUDA headers
#include <cufft.h>

// local headers
#include "nfft_adjoint.h"





FILE *out;
char fname[200];

char * cufftParseError(cufftResult_t r){
	char *message = (char *) malloc( 100 * sizeof(char));
	switch(r){
		case CUFFT_SUCCESS:
			sprintf(message, "The cuFFT operation was successful.");
			return message;
		case CUFFT_INVALID_PLAN:
			sprintf(message, "cuFFT was passed an invalid plan handle.");
			return message;
		case CUFFT_ALLOC_FAILED:
			sprintf(message, "cuFFT failed to allocate GPU or CPU memory.");
			return message;
		case CUFFT_INVALID_TYPE:
			sprintf(message, "CUFFT_INVALID_TYPE (no longer used)");
			return message;
		case CUFFT_INVALID_VALUE:
			sprintf(message, "User specified an invalid pointer or parameter");
			return message;
		case CUFFT_INTERNAL_ERROR:
			sprintf(message, "Driver or internal cuFFT library error.");
			return message;
		case CUFFT_EXEC_FAILED:
			sprintf(message, "Failed to execute an FFT on the GPU.");
			return message;
		case CUFFT_SETUP_FAILED:
			sprintf(message, "The cuFFT library failed to initialize.");
			return message;
		case CUFFT_INVALID_SIZE:
			sprintf(message, "User specified an invalid transform size.");
			return message;
		case CUFFT_UNALIGNED_DATA:
			sprintf(message, "CUFFT_UNALIGNED_DATA (no longer used).");
			return message;
		case CUFFT_INCOMPLETE_PARAMETER_LIST:
			sprintf(message, "Missing parameters in call.");
			return message;
		case CUFFT_INVALID_DEVICE:
			sprintf(message, "Execution of a plan was on different GPU than plan creation. ");
			return message;
		case CUFFT_PARSE_ERROR:
			sprintf(message, "Internal plan database error.");
			return message;
		case CUFFT_NO_WORKSPACE:
			sprintf(message, "No workspace has been provided prior to plan execution.");
			return message;
		default:
			sprintf(message, "DONT UNDERSTAND THE CUFFT ERROR CODE!! %d", r);
			return message;
	}
}

void checkCufftError(cufftResult_t r){
	if (r == CUFFT_SUCCESS) return;
	
	fprintf(stderr, "cuFFT ERROR: %s\n", cufftParseError(r));
	exit(r);
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
	
	
	if (!EQUALLY_SPACED) {
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
	
		sprintf(fname, "%s_gridded_data.dat", p->out_root);
		out = fopen(fname, "w");
		printComplex_d(p->g_f_hat, p->Ngrid, out);
		fclose(out);
	} else {
		// otherwise just copy data -> gridded data (same)
		checkCudaErrors(
			cudaMemcpy(	p->g_f_hat, 
						p->g_f_data, 
						p->Ndata * sizeof(Complex), 
						cudaMemcpyDeviceToDevice
					)
			);
	}

	
	
	
	if (!EQUALLY_SPACED) {
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

		sprintf(fname, "%s_gridded_filter.dat", p->out_root);
		out = fopen(fname, "w");
		printComplex_d(p->g_f_filter, p->Ngrid, out);
		fclose(out);
	} 
	

	LOG("planning cufftPlan");
	// make plan
	cufftHandle cuplan;

	#ifdef DOUBLE_PRECISION
	checkCufftError(
		cufftPlan1d(
		           &cuplan,
		           p->Ngrid * sizeof(Complex),
		           CUFFT_Z2Z,
		           1
		)
	);
	#else
	checkCufftError(
		cufftPlan1d(
		           &cuplan,
		           p->Ngrid * sizeof(Complex),
		           CUFFT_C2C,
		           1
		)
	);
	#endif

	checkCudaErrors(cudaDeviceSynchronize());


	
	LOG("doing FFT of gridded data.");
	// FFT(gridded data)
	#ifdef DOUBLE_PRECISION
	checkCufftError(
		cufftExecZ2Z(  
					cuplan,
		    		(fftComplex *)(p->g_f_hat),
		    		(fftComplex *)(p->g_f_hat),
					CUFFT_INVERSE
		)
	);

	#else
	checkCufftError(
		cufftExecC2C(  
					cuplan,
		        	(fftComplex *)(p->g_f_hat),
					(fftComplex *)(p->g_f_hat),
					CUFFT_INVERSE
		)
	);
	#endif
	
	checkCudaErrors(cudaDeviceSynchronize());
	
	sprintf(fname, "%s_FFT_raw_f_hat.dat", p->out_root);
	out = fopen(fname, "w");
	printComplex_d(p->g_f_hat, p->Ngrid, out);
	fclose(out);

	
	if(!EQUALLY_SPACED) {
		nblocks = p->Ngrid / BLOCK_SIZE;
		while (nblocks * BLOCK_SIZE < p->Ngrid) nblocks++;

		LOG("Normalizing");
		// normalize (eq. 11 in Greengard & Lee 2004)
		normalize <<< nblocks, BLOCK_SIZE >>>(
		    p->g_f_hat,
		    p->Ngrid,
		    p->fprops_device
		);
		
		checkCudaErrors(cudaDeviceSynchronize());
	}
	
	
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


