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