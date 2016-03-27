/*   cuna.h
 *   ======
 *
 *   Include for external projects looking to use CUNA (that don't
 *   explicitly use CUDA).
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
#ifndef CUNA_H
#define CUNA_H
#include <complex.h>
#include "cuna_typedefs.h"
/**
 * 
 * 
 * 
 */
void cunfft_adjoint(const dTyp *x, const dTyp *f, cTyp *fft, int n, 
                    int ng, unsigned int flags);

/**
 * 
 * 
 * 
 */
void cunfft_adjoint_from_plan( plan *p );

/**
 *
 *
 *
 */
void cunfft_adjoint_raw(const dTyp *x, const dTyp *f_data,
    Complex *f_hat, const int n, const int ng, const filter_properties *gpu_fprops);

void cunfft_adjoint_raw_async(const dTyp *x, const dTyp *f_data, 
    Complex *f_hat, const int n, const int ng, const filter_properties *gpu_fprops,
    cudaStream_t stream);

void cunfft_adjoint_raw_async_bootstrap(const dTyp *x, const dTyp *f_data, Complex *f_hat,
					const int n, const int ng, const int nbs, const filter_properties *gpu_fprops, cudaStream_t stream);
#endif
