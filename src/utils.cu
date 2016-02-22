#include "utils.h"
#include "filter.h"
#include <stdlib.h>




// Copies over a float array to Complex array
// TODO: Find a more efficient/sensible way to do this.
void copy_float_to_complex(float *a, Complex *b, size_t N){
	for (unsigned int i = 0; i < N; i++){
		b[i].x = a[i];
		b[i].y = 0;
	}
}

void set_x(float *x, size_t size){
	// ensures that x \in [0, 2pi)

	float range = x[size-1] - x[0];
	for(unsigned int i = 0; i < size; i++){
		x[i]-=x[0];
		x[i]/=range;
		x[i] *= 2 * PI;
	}
}

__device__
unsigned int
get_index(){ return blockIdx.x * BLOCK_SIZE + threadIdx.x; }


void free_filter_properties(filter_properties *f, free_type how_to_free){
	switch(how_to_free){
		case CUDA_FREE:
			checkCudaErrors(cudaFree(f->E1));
			checkCudaErrors(cudaFree(f->E2));
			checkCudaErrors(cudaFree(f->E3));
			checkCudaErrors(cudaFree(f));
			break;
		case CPU_FREE:
			free(f->E1);
			free(f->E2);
			free(f->E3);
			free(f);
			break;
		default:
			eprint("cannot understand free_type");
			break;
	}
}

void free_plan(plan *p){
	free(p->f_hat);
	free(p->x_data);
	free(p->f_data);
	free_filter_properties(p->fprops, CUDA_FREE);

	checkCudaErrors(cudaFree(p->g_f_hat));
	checkCudaErrors(cudaFree(p->g_f_filter));
	checkCudaErrors(cudaFree(p->g_f_data));
	checkCudaErrors(cudaFree(p->g_x_data));
}

__host__
void 
init_plan(
	plan 			*p, 
	dTyp 			*f, 
	dTyp 			*x, 
	unsigned int 	Ndata, 
	unsigned int 	Ngrid

){
	LOG("in init_plan -- mallocing for CPU");
	// Set 
	p->Ndata = Ndata;
	p->Ngrid = Ngrid;
	p->x_data = (dTyp *) malloc( Ndata * sizeof(dTyp));
	p->f_data = (dTyp *) malloc( Ngrid * sizeof(dTyp));

	LOG("memcpy x and f to plan");
	memcpy(p->x_data, x, Ndata * sizeof(dTyp));
	memcpy(p->f_data, f, Ndata * sizeof(dTyp));

	// Allocate GPU memory
	LOG("cudaMalloc -- p->g_f_data");
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_f_data), 
			p->Ndata * sizeof(Complex))
	);
	LOG("cudaMalloc -- p->g_x_data");
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_x_data), 
			p->Ndata * sizeof(dTyp))
	);
	LOG("cudaMalloc -- p->g_f_hat");
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_f_hat), 
			p->Ngrid * sizeof(Complex))
	);

	LOG("cudaMalloc -- p->g_f_filter");
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_f_filter), 
			p->Ngrid * sizeof(Complex))
	);

	LOG("copying f_data to f_data_complex");
	// "Cast" float array to Complex array
	Complex f_data_complex[p->Ndata];
	copy_float_to_complex(p->f_data, f_data_complex, p->Ndata);

	LOG("cudaMemcpy f_data_complex ==> p->g_f_data");
	// Copy f_j -> GPU
	checkCudaErrors(
		cudaMemcpy(p->g_f_data, f_data_complex, 
			p->Ndata * sizeof(float), cudaMemcpyHostToDevice)
	);

	LOG("cudaMemcpy p->x_data ==> p->g_x_data");
	// Copy x_j -> GPU
	checkCudaErrors(
		cudaMemcpy(p->g_x_data, p->x_data, 
			p->Ndata * sizeof(float), cudaMemcpyHostToDevice)
	);

	LOG("done here, calling set_filter_properties");
	// copy filter information + perform 
	// precomputation
	set_filter_properties(p);

}
