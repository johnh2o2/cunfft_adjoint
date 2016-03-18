/*   cuna_utils.h
 *   ============ 
 *   
 *   Misc. functions useful for other parts of the program 
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

#ifndef CUNA_UTILS_H
#define CUNA_UTILS_H

#include "cuna_typedefs.h"

// used to try and produce cuda overhead delay
__global__ void dummyKernel();

// launcher function for programs that don't explicitly use cuda
__host__ void launchDummyKernel();

// returns the next largest integer of the form 2^a where a \in (natural numbers)
__host__ unsigned int nextPowerOfTwo(unsigned int v);

// converts clock_t value into seconds
__host__ dTyp seconds(clock_t dt);

// converts real array to complex
__global__ void convertToComplex(const dTyp *a, Complex *c, const int N);

// Rescale X to [-1/2, 1/2)
__host__   void scale_x(dTyp *x, int size);

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
