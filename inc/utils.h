/*   utils.h
 *   =======   
 *   
 *   
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

#ifndef UTILS_
#define UTILS_

#include "typedefs.h"

// used to try and produce cuda overhead delay
__global__ void dummyKernel();

__host__ void launchDummyKernel();

__host__ unsigned int nextPowerOfTwo(unsigned int v);

__host__ void meanAndVariance(int n, const dTyp *y, dTyp *mean , dTyp *variance);

__device__ dTyp sign(dTyp a, dTyp b);

__device__ dTyp square(dTyp a);


// converts clock_t value into seconds
__host__ dTyp seconds(clock_t dt);

// generates unequal timing array
__host__ dTyp * generateRandomTimes(int N);

// generates a periodic signal
__host__ dTyp * generateSignal(dTyp *x, dTyp f, dTyp phi, int N);

// checks if any nans are in the fft
__host__ int countNans(Complex *fft, int N);

// Copies over a float array to Complex array
// TODO: Find a more efficient/sensible way to do this.
__host__   void copy_float_to_complex(dTyp *a, Complex *b, int N);

// converts real array into complex array
__host__   Complex* make_complex(dTyp *a, int N);

// Rescale X to [0, 2pi)
__host__   void scale_x(dTyp *x, int size);

// init/free plan
__host__   void init_plan( plan *p, const dTyp *x, const dTyp *f, 
                         int Ndata, int Ngrid, unsigned int plan_flags);
__host__   void free_plan( plan *p);


//OUTPUT UTILS
__host__   void print_plan(     plan *p);
__global__ void print_filter_props_d(filter_properties *f, int Ndata);
// GPU arrays
__host__   void printComplex_d( Complex *a, int N, FILE *out);
__host__   void printReal_d(    dTyp    *a, int N, FILE *out);
// CPU arrays
__host__   void printComplex(   Complex *a, int N, FILE *out);
__host__   void printReal(      dTyp    *a, int N, FILE *out);


// CUDA doesn't have a native atomic function if the variables are
// double precision, so we add an override here if we're doing double prec.
#ifdef DOUBLE_PRECISION
__device__ double atomicAdd(double* address, double val);
#endif

#endif
