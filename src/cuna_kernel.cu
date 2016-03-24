/*   cuna_kernel.cu
 *   ==============   
 *   
 *   Implementation of the adjoint NFFT operation
 * 
 *   (c) John Hoffman 2016
 * 
 *   This file is part of CUNA
 *
 *   CUNA is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CUNA is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CUNA.  If not, see <http://www.gnu.org/licenses/>.
 */

// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CUDA headers
#include <cufft.h>

// local headers
#include "cuna.h"
#include "cuna_utils.h"
#include "cuna_filter.h"
#include "cuna_gridding.h"
#include "cufft_utils.h"

FILE *out;
char fname[200];

#ifdef DOUBLE_PRECISION
#define CUFFT_EXEC_CALL cufftExecZ2Z
#define CUFFT_TRANSFORM_TYPE CUFFT_Z2Z
#else
#define CUFFT_EXEC_CALL cufftExecC2C
#define CUFFT_TRANSFORM_TYPE CUFFT_C2C
#endif


__host__ void 
performGridding(plan *p) {

    int nblocks;
    nblocks = p->Ndata / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < p->Ndata) nblocks++;
    
    // unequally spaced data -> equally spaced grid
    fast_gridding <<< nblocks, BLOCK_SIZE >>>
          ( p->g_f_data, p->g_f_hat, p->g_x_data, p->Ngrid,
            p->Ndata, p->fprops_device );

    if(p->flags & CALCULATE_WINDOW_FUNCTION) {
        // unequally spaced data -> equally spaced grid
        fast_gridding <<< nblocks, BLOCK_SIZE >>>
              ( NULL, p->g_f_hat_win, p->g_x_data,
                p->Ngrid, p->Ndata, p->fprops_device );
    }

}

__host__ void 
performFFTs(plan *p) {
    int nblocks;
    nblocks = p->Ngrid / BLOCK_SIZE;
    while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++; 

    // make plan
    cufftHandle cuplan; 
    checkCufftError(
        cufftPlan1d( &cuplan, p->Ngrid, CUFFT_TRANSFORM_TYPE, 1)
    );

    // FFT(gridded data)
    checkCufftError(
        CUFFT_EXEC_CALL( cuplan, p->g_f_hat, p->g_f_hat, CUFFT_INVERSE )
    );
    
    
    if (p->flags & CALCULATE_WINDOW_FUNCTION) {
        
        // FFT(gridded data)
        checkCufftError(
            CUFFT_EXEC_CALL( cuplan, p->g_f_hat_win,p->g_f_hat_win,
                             CUFFT_INVERSE)
        );
    }
    cufftDestroy(cuplan);
}

__host__ void 
normalizeResults(plan *p) {
    int nblocks;
    nblocks = p->Ngrid / BLOCK_SIZE;
    while(nblocks * BLOCK_SIZE < p->Ngrid) nblocks++; 

    // normalize (eq. 11 in Greengard & Lee 2004)
    normalize <<< nblocks, BLOCK_SIZE >>>
          ( p->g_f_hat, p->Ngrid, p->fprops_device );

    if(p->flags & CALCULATE_WINDOW_FUNCTION) {
        // normalize (eq. 11 in Greengard & Lee 2004)
        normalize <<< nblocks, BLOCK_SIZE >>>
              ( p->g_f_hat_win, p->Ngrid, p->fprops_device );
    }
}


__host__ void 
copyResultsToCPU(plan *p) {
    // Transfer back to device!
    checkCudaErrors(
        cudaMemcpy( p->f_hat, p->g_f_hat, p->Ngrid * sizeof(Complex),
                    cudaMemcpyDeviceToHost )
    );
    if(p->flags & CALCULATE_WINDOW_FUNCTION) {
        // Transfer back to device!
        checkCudaErrors(
            cudaMemcpy( p->f_hat_win, p->g_f_hat_win,
                        p->Ngrid * sizeof(Complex), cudaMemcpyDeviceToHost)
        );
    }
}

// computes the adjoint NFFT and stores this in plan->f_hat
__host__ void 
cunfft_adjoint_from_plan(plan *p) {
    performGridding(p);
    performFFTs(p);
    normalizeResults(p);
    if ( !(p->flags | DONT_TRANSFER_TO_CPU)){
       copyResultsToCPU(p);
    }
}

// x, f_data, f_grid, and f_hat should all be allocated on the GPU
void
cunfft_adjoint_raw(const dTyp *x, const dTyp *f_data, Complex *f_hat, 
		const int n, const int ng, const filter_properties *gpu_fprops) {

    int nblocks;

    // set number of blocks & threads
    nblocks = n / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < n) nblocks++;
    
    // unequally spaced data -> equally spaced grid
    fast_gridding <<< nblocks, BLOCK_SIZE >>> 
                            ( f_data, f_hat, x, ng, n, gpu_fprops );

    
    // resize nblocks
    nblocks = ng / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < ng) nblocks++; 

    // make plan
    cufftHandle cuplan; 
    checkCufftError(cufftPlan1d( &cuplan, ng, CUFFT_TRANSFORM_TYPE, 1));

    // FFT(gridded data)
    checkCufftError(CUFFT_EXEC_CALL( cuplan, f_hat, f_hat, CUFFT_INVERSE ));

    // wait for process to finish
    checkCudaErrors(cudaThreadSynchronize());
  
    // destroy plan
    cufftDestroy(cuplan);

    // normalize (eq. 11 in Greengard & Lee 2004)
    normalize <<< nblocks, BLOCK_SIZE >>>( f_hat, ng, gpu_fprops );

}

// ASYNCHRONOUS version of cunfft_adjoint_raw
// x, f_data, f_grid, and f_hat should all be allocated with cudaMalloc or cudaHostMalloc
void
cunfft_adjoint_raw_async(const dTyp *x, const dTyp *f_data, Complex *f_hat, 
	const int n, const int ng, const filter_properties *gpu_fprops, 
   	cudaStream_t stream) {

    int nblocks;

    // set number of blocks & threads
    nblocks = n / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < n) nblocks++;

    // unequally spaced data -> equally spaced grid
    fast_gridding <<< nblocks, BLOCK_SIZE, 0, stream >>>
                            ( f_data, f_hat, x, ng, n, gpu_fprops );

    // SYNCHRONIZE
    checkCudaErrors(cudaStreamSynchronize(stream));

    // make plan
    cufftHandle cuplan;
    checkCufftError(cufftPlan1d( &cuplan, ng, CUFFT_TRANSFORM_TYPE, 1));

    // set the CUDA stream
    checkCufftError(cufftSetStream(cuplan, stream));

    // FFT(gridded data)
    checkCufftError(CUFFT_EXEC_CALL( cuplan, f_hat, f_hat, CUFFT_INVERSE ));

    // SYNCHRONIZE
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    // destroy plan
    cufftDestroy(cuplan);

    // set number of blocks & threads
    nblocks = ng / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < ng) nblocks++;

    // normalize (eq. 11 in Greengard & Lee 2004)
    normalize <<< nblocks, BLOCK_SIZE, 0, stream >>>( f_hat, ng, gpu_fprops );

    // SYNCHRONIZE
    checkCudaErrors(cudaStreamSynchronize(stream));
}

__host__
int
get_max(int *x, int n) {
    int m = x[0];
    for(int i = 0; i < n; i++) 
        if (x[i] > m)
	    m = x[i];

    return m;
}
/*
// ASYNCHRONOUS, BATCHED version of cunfft_adjoint_raw
// x, f_data, f_grid, and f_hat should all be allocated with cudaMalloc
// f_hat should be allocated to be max(ng) * nlc * sizeof(Complex)
// 
void
cunfft_adjoint_raw_batched_async(const dTyp *x, const dTyp *f_data, Complex *f_hat, 
	const int *n, const int *ng, const int nlc, const filter_properties **gpu_fprops, 
   	cudaStream_t stream) {

    // find max(ng)
    int idist = ng[0], nobs_tot = 0, ng_tot=0, nblocks;
    for(int i = 0; i < nlc; i++) {
	    if (ng[i] > idist) idist = ng[i];
        nobs_tot +=  n[i];
        ng_tot   += ng[i];
    }

    // set number of blocks & threads
    nblocks = nobs_tot / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < nobs_tot) nblocks++;

    // unequally spaced data -> equally spaced grid
    fast_gridding_batched <<< nblocks, BLOCK_SIZE, 0, stream >>>
                            ( f_data, f_hat, x, ng, n, nlc, idist, gpu_fprops );
    
    // SYNCHRONIZE
    checkCudaErrors(cudaStreamSynchronize(stream));

    // make batched plan
    cufftHandle cuplan; 
    int dims[1] = { idist };
    checkCufftError(cufftPlanMany( &cuplan, 1, dims, NULL, 1, idist, NULL, 1, CUFFT_TRANSFORM_TYPE, nlc));

    // set the CUDA stream
    checkCufftError(cufftSetStream(cuplan, stream));

    // FFT(gridded data)
    checkCufftError(CUFFT_EXEC_CALL( cuplan, f_hat, f_hat, CUFFT_INVERSE ));

    // SYNCHRONIZE
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    // destroy plan
    cufftDestroy(cuplan);

    // set number of blocks & threads
    nblocks = ng_tot / BLOCK_SIZE;
    while (nblocks * BLOCK_SIZE < ng_tot) nblocks++;

    // normalize (eq. 11 in Greengard & Lee 2004)
    normalize_batch <<< nblocks, BLOCK_SIZE, 0, stream >>>( f_hat, ng, nlc, idist, gpu_fprops );

    // SYNCHRONIZE
    checkCudaErrors(cudaStreamSynchronize(stream));
}
*/


