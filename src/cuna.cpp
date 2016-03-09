/*   cuna.cpp
 *   ========
 *
 *   Wrapper for the CUNA operations
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

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <cuda.h>

#include "cuna.h"
#include "cuna_typedefs.h"
#include "cuna_utils.h"


// wrapper for the adjoint NFFT
void cunfft_adjoint(const dTyp *x, const dTyp *f, cTyp *fft, int n, int ng, unsigned int flags) {
	plan *p = (plan *) malloc(sizeof(plan));

	init_plan(p, x, f, n, ng, flags);

	cunfft_adjoint_from_plan(p);

	for (int i = 0; i < ng; i++)
		fft[i] = p->f_hat[i].x + _Complex_I * p->f_hat[i].y;

	free_plan(p);
}


