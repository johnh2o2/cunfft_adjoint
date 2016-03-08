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

char * cufftParseError(cufftResult_t r);
void checkCufftError(cufftResult_t r);

#ifdef DOUBLE_PRECISION
#define CUFFT_EXEC_CALL cufftExecZ2Z
#define CUFFT_TRANSFORM_TYPE CUFFT_Z2Z

#else
#define CUFFT_EXEC_CALL cufftExecC2C
#define CUFFT_TRANSFORM_TYPE CUFFT_C2C

#endif

__global__
void
// shifts in k-space by -Ngrid/2 to be consistent with nfft3
convertToComplex(dTyp *a, Complex *c, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		c[i].x = a[i];
		c[i].y = 0;
    }
}


__host__ void performGridding(plan *p) {

    int nblocks;
    nblocks = p->Ndata / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < p->Ndata) nblocks++;
    
    LOG("about to do fast_gridding");
    // unequally spaced data -> equally spaced grid
    fast_gridding <<< nblocks, BLOCK_SIZE >>>(
                       p->g_f_data,
                       p->g_f_grid,
                       p->g_x_data,
                       p->Ngrid,
                       p->Ndata,
                       p->fprops_device
    );
    if(p->flags & CALCULATE_WINDOW_FUNCTION) {
        LOG("about to do fast_gridding (WINDOW)");
        // unequally spaced data -> equally spaced grid
        fast_gridding <<< nblocks, BLOCK_SIZE >>>(
                           NULL,
                           p->g_f_grid_win,
                           p->g_x_data,
                           p->Ngrid,
                           p->Ndata,
                           p->fprops_device
        );
    }

    if (p->flags & OUTPUT_INTERMEDIATE) {
        LOG("writing gridded_data");
        out = fopen("gridded_data.dat", "w");
        printReal_d(p->g_f_grid, p->Ngrid, out);
        fclose(out);
        if (p->flags & CALCULATE_WINDOW_FUNCTION) {
            LOG("writing gridded_data (WINDOW)");
            out = fopen("gridded_data_window.dat", "w");
            printReal_d(p->g_f_grid_win, p->Ngrid, out);
            fclose(out);
        }
    }
}

__host__ void transferGridResults(plan *p) {

    int nblocks;
    nblocks = p->Ngrid / BLOCK_SIZE;
    while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++; 

    LOG("converting g_f_grid to (Complex) g_f_hat");
    convertToComplex <<< nblocks, BLOCK_SIZE >>>
                   (p->g_f_grid, p->g_f_hat, p->Ngrid);

    if (p->flags & CALCULATE_WINDOW_FUNCTION) { 
        LOG("converting g_f_grid_win to (Complex) g_f_hat_win (WINDOW)");
        convertToComplex <<< nblocks, BLOCK_SIZE >>>
                   (p->g_f_grid_win, p->g_f_hat_win, p->Ngrid);
    }
}

__host__ void performFFTs(plan *p) {
    int nblocks;
    nblocks = p->Ngrid / BLOCK_SIZE;
    while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++; 

    LOG("calling cufftPlan1d");
    
    // make plan
    cufftHandle cuplan; 
    checkCufftError(
        cufftPlan1d(
                   &cuplan,
                   p->Ngrid,
                   CUFFT_TRANSFORM_TYPE,
                   1
        )
    );
    //LOG("synchronizing the device.");
    //checkCudaErrors(cudaDeviceSynchronize());

    LOG("doing FFT of gridded data.");
    // FFT(gridded data)
    checkCufftError(
        CUFFT_EXEC_CALL(  
                    cuplan,
                    p->g_f_hat,
                    p->g_f_hat,
                    CUFFT_INVERSE
        )
    );
    
    //LOG("synchronizing the device.");
    //checkCudaErrors(cudaDeviceSynchronize());
    
    if (p->flags & OUTPUT_INTERMEDIATE) { 
        LOG("outputting raw fft of gridded data.");
        out = fopen("FFT_raw_f_hat.dat", "w");
        printComplex_d(p->g_f_hat, p->Ngrid, out);
        fclose(out);
    }

    if (p->flags & CALCULATE_WINDOW_FUNCTION) {
        
        LOG("doing FFT of gridded data. (WINDOW)");
        // FFT(gridded data)
        checkCufftError(
            CUFFT_EXEC_CALL(  
                        cuplan,
                        p->g_f_hat_win,
                        p->g_f_hat_win,
                        CUFFT_INVERSE
            )
        );

        if (p->flags & OUTPUT_INTERMEDIATE) { 
            LOG("outputting raw fft of gridded data. (WINDOW)");
            out = fopen("FFT_raw_f_hat_win.dat", "w");
            printComplex_d(p->g_f_hat_win, p->Ngrid, out);
            fclose(out);
        }

    }
    LOG("destroying cufft plan");
    cufftDestroy(cuplan);
}

__host__ void normalizeResults(plan *p) {
    int nblocks;
    nblocks = p->Ngrid / BLOCK_SIZE;
    while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++; 

    LOG("Normalizing");
    // normalize (eq. 11 in Greengard & Lee 2004)
    normalize <<< nblocks, BLOCK_SIZE >>>(
            p->g_f_hat,
            p->Ngrid,
            p->fprops_device
    );

    if(p->flags & CALCULATE_WINDOW_FUNCTION) {
        LOG("Normalizing (WINDOW)");
        // normalize (eq. 11 in Greengard & Lee 2004)
        normalize <<< nblocks, BLOCK_SIZE >>>(
                p->g_f_hat_win,
                p->Ngrid,
                p->fprops_device
        );
    }
}


__host__ void copyResultsToCPU(plan *p) {
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
    if(p->flags & CALCULATE_WINDOW_FUNCTION) {
        LOG("Transferring data back to device (WINDOW)");
    
        // Transfer back to device!
        checkCudaErrors(
            cudaMemcpy(
                p->f_hat_win,
                p->g_f_hat_win,
                p->Ngrid * sizeof(Complex),
                cudaMemcpyDeviceToHost
            )
        );
    }
}

#define timeCommand(command)\
   if(p->flags | PRINT_TIMING) \
       start=clock(); \
   command;\
   if(p->flags | PRINT_TIMING) \
       fprintf(stderr, "  NFFT_ADJOINT.CU: %-20s : %.4e(s)\n", #command, seconds(clock() - start))

// computes the adjoint NFFT and stores this in plan->f_hat
__host__ void cuda_nfft_adjoint(plan *p) {
    clock_t start;
    timeCommand(performGridding(p));
    timeCommand(transferGridResults(p));
    timeCommand(performFFTs(p));
    timeCommand(normalizeResults(p));
    if ( !(p->flags | DONT_TRANSFER_TO_CPU)){
       timeCommand(copyResultsToCPU(p));
    }
}

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

/*
// used for debugging.
__host__
Complex *
testCufft(dTyp *x, dTyp *f, int N){

	// Allocate GPU memory
	Complex *d_x, *d_f;
	checkCudaErrors(
		cudaMalloc((void **) &d_x, N * sizeof(Complex))
	);

	checkCudaErrors(
		cudaMalloc((void **) &d_x, N * sizeof(Complex))
	);

	// Convert CPU data to complex
	Complex *xC = make_complex(x, N);
	Complex *fC = make_complex(f, N);

	// Copy data to GPU
	checkCudaErrors(
		cudaMemcpy(d_x, xC, N * sizeof(Complex), cudaMemcpyHostToDevice)
	);
	checkCudaErrors(
		cudaMemcpy(d_f, fC, N * sizeof(Complex), cudaMemcpyHostToDevice)
	);

	// Plan FFT
	cufftHandle p;
	cufftPlan1d( &p, N, CUFFT_C2C, 1 );

	// Allocate space for FFT
	Complex *fft, *d_fft;

	// CPU
	fft = (Complex *)malloc(N * sizeof(Complex));

	// GPU
	checkCudaErrors(
		cudaMalloc((void **)&d_fft, N * sizeof(Complex))
	);

	/////// DO FFT ///////
	cufftExecC2C( p, d_f, d_fft, CUFFT_INVERSE );

	// Copy results back to CPU
	checkCudaErrors(
		cudaMemcpy(fft, d_fft, N * sizeof(Complex), cudaMemcpyDeviceToHost)
	);

	return fft;
}
*/
