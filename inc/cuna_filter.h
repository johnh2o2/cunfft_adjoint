/*   cuna_filter.h
 *   =============
 *   
 *   Defines API for implementing a filter
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

#ifndef CUNA_FILTER_H
#define CUNA_FILTER_H

#include "cuna_typedefs.h"

/**
 *
 *
 *
 */
__host__ void 
set_filter_properties( plan *p ); 

__host__ void
generate_pinned_filter_properties(const dTyp *x, const int n, const int ng,
        filter_properties *h_fprops, filter_properties *d_fprops, 
	cudaStream_t stream);

/**
 *
 *
 *
 */
__host__ void
generate_filter_properties(const dTyp *x, const int n, const int ng, 
	filter_properties **fprops_host, filter_properties **fprops_device);

/**
 *
 *
 *
 */
__host__ void
free_filter_properties(filter_properties *d_fp, filter_properties *fp);

/**
 * 
 * 
 * 
 */
__device__ dTyp 
filter( const int j_data, const int i_grid, const int m, const filter_properties *f);

/**
 * 
 * 
 * 
 */
__global__ void 
normalize( Complex *f_hat, const int Ngrid, const filter_properties *f );


__global__ void
normalize_bootstrap( Complex *f_hat, const int Ngrid, const int Nbootstrap, 
                     const filter_properties *f);
#endif
