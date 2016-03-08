/** \file ls.c
 * Implementation of LombScargle().
 *
 * \author B. Leroy
 *
 * This file is part of nfftls.
 *
 * Nfftls is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Nfftls is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with nfftls.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright (C) 2012 by B. Leroy
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuComplex.h>

#include "nfft_adjoint.h"
#include "typedefs.h"



__global__
void
convertToLSP( const Complex *sp, const Complex *win, dTyp var, int m, int npts, dTyp *lsp) {

  int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if ( j < m ) {
    Complex z1 = sp[j];
    Complex z2 = win[j];
    dTyp hypo = cuAbs(z2);
    dTyp hc2wt = 0.5 * cuImag(z2) / hypo;
    dTyp hs2wt = 0.5 * cuReal(z2) / hypo;
    dTyp cwt = cuSqrt(0.5 + hc2wt);
    dTyp swt = sign(cuSqrt(0.5 - hc2wt), hs2wt);
    dTyp den = 0.5 * npts + hc2wt * cuReal(z2) + hs2wt * cuImag(z2);
    dTyp cterm = square(cwt * cuReal(z1) + swt * cuImag(z1)) / den;
    dTyp sterm = square(cwt * cuImag(z1) - swt * cuReal(z1)) / (npts - den);

    lsp[j] = (cterm + sterm) / (2 * var);
  }
}

#define EPSILON 1e-5
__host__
dTyp *
scaleTobs(const dTyp *tobs, int npts, dTyp oversampling) {

  // clone data
  dTyp * t = (dTyp *)malloc(npts * sizeof(dTyp));

  // now transform t -> [-1/2, 1/2)
  dTyp tmax  = tobs[npts - 1];
  dTyp tmin  = tobs[0];

  dTyp range = (tmax - tmin) * oversampling;
  dTyp a     = 0.5 - EPSILON;

  for(int i = 0; i < npts; i++) 
    t[i] = 2 * a * (tobs[i] - tmin)/range - a;
  
  return t;
}


__host__
dTyp *
scaleYobs(const dTyp *yobs, int npts, dTyp *var) {
  dTyp avg;
  meanAndVariance(npts, yobs, &avg, var);
  
  dTyp *y = (dTyp *)malloc(npts * sizeof(dTyp));
  for(int i = 0; i < npts; i++)
    y[i] = yobs[i] - avg;
  
  return y;
}

#define START_TIMER if(flags | PRINT_TIMING) start = clock()
#define STOP_TIMER(name)  if(flags | PRINT_TIMING)\
      fprintf(stderr, "[ lombScargle ] %-20s : %.4e(s)\n", #name, seconds(clock() - start))


__host__ 
dTyp *
lombScargle(const dTyp *tobs, const dTyp *yobs, int npts, 
            dTyp over, dTyp hifac, int * ng, unsigned int flags) {
  clock_t start;
  // allocate memory for NFFT
  // Note: the LSP calculations can be done more efficiently by
  //       bypassing the cuda-nfft plan system 
  //       since there are redundant transfers/allocations
  plan *p  = (plan *)malloc(sizeof(plan));

  // size of LSP
  unsigned int NG = (unsigned int) floor(0.5 * npts * over * hifac);

  // round to the next power of two
  *ng = (int) nextPowerOfTwo(NG);

  // correct the "oversampling" parameter accordingly
  over *= ((float) (*ng))/((float) NG);

  // calculate number of CUDA blocks we need
  int nblocks = *ng / BLOCK_SIZE;
  while (nblocks * BLOCK_SIZE < *ng) nblocks++;

  // scale t and y (zero mean, t \in [-1/2, 1/2))
  dTyp var;
  dTyp *t = scaleTobs(tobs, npts, over);
  dTyp *y = scaleYobs(yobs, npts, &var);

  // initialize plans with scaled arrays
  START_TIMER;
  init_plan(p , t, y   , npts,     (*ng), 
           CALCULATE_WINDOW_FUNCTION | DONT_TRANSFER_TO_CPU | flags);
  STOP_TIMER("init_plan");

  // evaluate NFFT for window + signal
  START_TIMER;
  cuda_nfft_adjoint(p);
  STOP_TIMER("cuda_nfft_adjoint");

  // allocate GPU memory for lsp
  dTyp *d_lsp;
  checkCudaErrors(
    cudaMalloc((void **) &d_lsp, (*ng) * sizeof(dTyp))
  );
  
  // convert to LSP
  START_TIMER;
  convertToLSP <<< nblocks, BLOCK_SIZE >>> 
           (p->g_f_hat, p->g_f_hat_win, var, (*ng), npts, d_lsp);
  STOP_TIMER("convertToLSP");

  // Copy the results back to CPU memory
  START_TIMER;
  dTyp *lsp = (dTyp *) malloc( (*ng) * sizeof(dTyp) );
  checkCudaErrors(
    cudaMemcpy(lsp, d_lsp, (*ng) * sizeof(dTyp), cudaMemcpyDeviceToHost)
  )
  STOP_TIMER("malloc + memcpy lsp");

  return lsp;

}

/**
 * Returns the probability that a peak of a given power
 * appears in the periodogram when the signal is white
 * Gaussian noise.
 *
 * \param Pn the power in the periodogram.
 * \param npts the number of samples.
 * \param nfreqs the number of frequencies in the periodogram.
 * \param over the oversampling factor.
 *
 * \note This is the expression proposed by A. Schwarzenberg-Czerny
 * (MNRAS 1998, 301, 831), but without explicitely using modified
 * Bessel functions.
 * \note The number of effective independent frequencies, effm,
 * is the rough estimate suggested in Numerical Recipes. 
 */
__host__
dTyp 
probability(dTyp Pn, int npts, int nfreqs, dTyp over)
{
  dTyp effm = 2.0 * nfreqs / over;
  dTyp Ix = 1.0 - pow(1 - 2 * Pn / npts, 0.5 * (npts - 3));

  dTyp proba = 1 - pow(Ix, effm);
  
  return proba;
}
