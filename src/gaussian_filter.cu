/* Implements the Gaussian filter for the NFFT
 *
 * (c) John Hoffman 2016
 * jah5@princeton.edu
 * 
 */

#include "typedefs.h"
#include "filter.h"
#include "utils.h"

__host__
// pre-computes values for the filter
void set_filter_properties(plan *p){

	// nblocks x BLOCK_SIZE threads
	unsigned int nblocks = p->Ngrid / BLOCK_SIZE;

	// Ensures that we have enough threads!
	while (nblocks * BLOCK_SIZE < n) nblocks++;

	// malloc memory for filter_properties (on GPU)
	filter_properties *f;
	checkCudaErrors(cudaMalloc((void **) &f, sizeof(filter_properties *) ));

	checkCudaErrors(cudaMalloc((void **) &(f->E1), p->Ndata * sizeof(dTyp) ));
	checkCudaErrors(cudaMalloc((void **) &(f->E2), p->Ndata * sizeof(dTyp) ));
	checkCudaErrors(cudaMalloc((void **) &(f->E3), p->filter_radius * sizeof(dTyp) ));


	// R    :  is the oversampling factor
	dTyp R = ((dTyp) p->Ngrid) / p->Ndata;

	// tau  :  is the characteristic length scale for the filter 
	//         (not to be confused with the filter_radius)
	dTyp tau = (1.0 / (p->Ngrid * p->Ngrid)) * ( PI / (R* (R - 0.5)) ) * p->filter_radius;

	// Copy over R and tau to GPU
	checkCudaErrors(cudaMemcpy(&(f->tau),&tau, sizeof(dTyp), cudaMemcpyHostToDevice ));
	checkCudaErrors(cudaMemcpy(&(f->filter_radius),&(p->filter_radius), 
		sizeof(int), cudaMemcpyHostToDevice ));

	// Precompute E1, E2, E3 on GPU
	set_gpu_filter_properties<<<nblocks, BLOCK_SIZE>>>(f, p->x_data, p->Ngrid, p->Ndata);

	// Set plan's filter_properties pointer to this particular filter properties object
	p->fprops = f;

}



/////////////////////////////////////////////
//  Uses GPU to precompute relevant values //
/////////////////////////////////////////////
__global__
void
set_gpu_filter_properties( filter_properties *f, dTyp *x, const unsigned int Ngrid, const unsigned int Ndata ){
	size_t i = get_index();
	if ( i < Ndata){
		unsigned int m = i * Ngrid / Ndata;
		dTyp eps = x[i] - 2 * PI * m / Ngrid;
		f->E1[i] = expf(- eps * eps / (4 * f->tau));
		f->E2[i] = expf( eps * PI / (Ngrid * tau)); 
	}
	if ( i < f->filter_radius){
		dTyp a = PI * PI * i * i / (Ngrid * Ngrid);
		f->E3[i] = expf( -a / f->tau);
	}
	
}

__device__
dTyp
filter ( const unsigned int j_data, const unsigned int i_grid, const int m , filter_properties *f){
	unsigned int mp;
	if (m < 0) mp = -m;
	else mp = m; 
	return f->E1[j_data] * powf(f->E2[j_data], m) * f->E3[mp];
}

__global__
void
normalize(Complex *f_hat, unsigned int Ngrid){
	unsigned int i = get_index();
	int k;
	if ( i < Ngrid ){
		k = i - Ngrid/2;
		f_hat[i] *= sqrt(PI/g_fprops->tau) * expf(k * k * g_fprops->tau);
	}
}
