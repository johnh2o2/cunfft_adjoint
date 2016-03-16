/*   cuna_typedefs.h
 *   ===============
 *   
 *   Contains global variables and type definitions for the rest of CUNA
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
#ifndef CUNA_TYPEDEFS_H
#define CUNA_TYPEDEFS_H

// standard headers (needed for print operations, etc)
#include <stdlib.h>
#include <stdio.h>

// CUDA specific headers
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cufft.h>


#define DEBUGSTREAM stderr

#ifdef DOUBLE_PRECISION
  #define dTyp double
  #define cTyp double complex
  #define Complex cufftDoubleComplex 
  #define cuComplexDivide cuCdiv
  #define cuReal cuCreal
  #define cuImag cuCimag
  #define cuAbs  cuCabs
  #define cuSqrt sqrt
  #define absoluteValueReal fabs
#else
  #define dTyp float
  #define cTyp float complex
  #define Complex cufftComplex
  #define cuComplexDivide cuCdivf
  #define cuReal cuCrealf
  #define cuImag cuCimagf
  #define cuAbs  cuCabsf
  #define cuSqrt sqrtf
  #define absoluteValueReal fabsf
#endif

// Never know the safest way to use PI (I'm guessing it's in math.h, but why not confuse people, right?)
#define PI 3.14159265358979323846

// Error printing
#define eprint(...) \
    fprintf(stderr, "[%-10s] %-30s L[%-5d]: ", "ERROR", __FILE__, __LINE__);\
    fprintf(stderr, __VA_ARGS__);\
    exit(EXIT_FAILURE)

// DEBUG flag turns on the "log" messages to DEBUGSTREAM (default is stderr)
#ifdef DEBUG
    #define LOG(msg)	fprintf(DEBUGSTREAM, "[%-10s] %-30s L[%-5d]: %s\n", "OK", __FILE__, __LINE__, msg)
#else
    #define LOG(msg) 
#endif


// This may also be implemented somewhere in CUDA, but this ensures that it exists and we can
// customize it ourselves. Pulled this from somewhere on StackExchange, can't find the original post!!
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// inline function for printing cuda errors
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA ERROR %-24s L[%-5d]: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
}

// FILTER PROPERTIES struct
typedef struct {
    // shape parameter, normalization factor
    dTyp b, normfac;

    // maximum cutoff for smoothing
    int filter_radius;

    // precomputed values
    dTyp *E1;
    dTyp *E2;
    dTyp *E3;

    // NOTE: these variables are unique to the Gaussian filter right now, 
    // I'm sure there's a more sensible way to define this here in
    // typedefs.h so that the user doesn't have to edit this file
    // when implementing some other filter.
} filter_properties;

typedef enum { 
  NO_FLAGS = 0x01,
  OUTPUT_INTERMEDIATE = 0x02,
  CALCULATE_WINDOW_FUNCTION = 0x04,
  DONT_TRANSFER_TO_CPU = 0x08,
  PRINT_TIMING = 0x10
} FLAGS;

// Our internal PLAN datatype
typedef struct {
    // CPU variables
    dTyp *x_data, *f_data;
    Complex *f_hat, *f_hat_win;

    // GPU variables
    Complex *g_f_hat, *g_f_hat_win;
    dTyp *g_x_data, *g_f_data, *g_f_grid, *g_f_grid_win;
 
    // optional directions
    unsigned int flags;

    // size of incoming data array
    int Ndata, 

    // size of grid to smooth data onto
    Ngrid, 

    // same as above
    filter_radius;

    // two filter properties structs here.
    // [CPU]fprops_host   { [GPU]E1, E2, E3, [CPU]tau, filter_radius },
    // [GPU]fprops_device { [GPU] ^,  ^,  ^, [GPU]tau, filter_radius } 
    filter_properties *fprops_host, *fprops_device;

    // tag to add onto output filenames (unit testing)
    //char out_root[100];

} plan;

// function to free plan memory
__host__ void free_plan(plan *p);

// function to initialize plan
__host__ void init_plan( plan *p, const dTyp *x, const dTyp *f,
                         int Ndata, int Ngrid, unsigned int plan_flags);
#endif
