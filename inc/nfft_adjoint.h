/*   nfft_adjoint.h
 *   ==============   
 *   
 *   (Simple) API for nfft adjoint operation
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

#ifndef NFFT_ADJOINT_
#define NFFT_ADJOINT_

#include "typedefs.h"
#include "filter.h"
#include "utils.h"
#include "adjoint_kernel.h"


__host__
void 
cuda_nfft_adjoint(
	plan 			*p

);

#endif