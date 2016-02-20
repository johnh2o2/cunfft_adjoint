#include "typedefs.h"
#include "utils.h"
#include <stdlib.h>

// Copies over a float array to Complex array
// TODO: Find a more efficient/sensible way to do this.
void copy_float_to_complex(float *a, Complex *b, size_t N){
	for (unsigned int i = 0; i < N; i++){
		b[i].x = a[i];
		b[i].y = 0;
	}
}


// Allocates and transfers plan data to GPU memory
void copy_data_to_gpu(plan *p){
	
	// Allocate GPU memory
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_f_data), 
			p->Ndata * sizeof(Complex))
	);
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_x_data), 
			p->Ndata * sizeof(float))
	);
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_f_hat), 
			p->Ngrid * sizeof(Complex))
	);
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_f_filter), 
			p->Ngrid * sizeof(Complex))
	);

	// "Cast" float array to Complex array
	Complex f_data_complex[p->Ndata];
	copy_float_to_complex(p->f_data, f_data_complex, p->Ndata);

	// Copy f_j -> GPU
	checkCudaErrors(
		cudaMemcpy(p->g_f_data, f_data_complex, 
			p->Ndata * sizeof(float), cudaMemcpyHostToDevice)
	);

	// Copy x_j -> GPU
	checkCudaErrors(
		cudaMemcpy(p->g_x_data, p->x_data, 
			p->Ndata * sizeof(float), cudaMemcpyHostToDevice)
	);
	

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

__global__
size_t
get_index(){
	return blockIdx.x * BLOCK_SIZE + threadIdx.x;
}


