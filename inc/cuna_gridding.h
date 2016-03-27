/*   cuna_gridding.h
 *   ===============   
 *   
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

#ifndef CUNA_GRIDDING_H
#define CUNA_GRIDDING_H

#include "cuna_typedefs.h"

/**
 *
 *
 */
__global__ void 
fast_gridding( const dTyp *f_data, Complex *f_hat, const dTyp *x_data, const int Ngrid, 
               const int Ndata, const filter_properties *fprops );

__global__ void 
fast_gridding_bootstrap( const dTyp *f_data, Complex *f_hat, const dTyp *x_data, const int Ngrid, 
               const int Ndata, const int Nbootstraps, const filter_properties *fprops,
	       const unsigned int seed );

#endif
