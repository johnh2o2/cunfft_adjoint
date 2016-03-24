/*   utils.cu
 *   ========  
 *   
 *   Misc. functions useful to the rest of the program 
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

#include "cuna_utils.h"
#include "cuna_filter.h"

#ifdef DOUBLE_PRECISION

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif

__global__ void
convertToComplex(const dTyp *a, Complex *c, const int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		c[i].x = a[i];
		c[i].y = 0;
    }
}

__host__ unsigned int 
nextPowerOfTwo(unsigned int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

__global__ void 
dummyKernel() { 
	int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
}

__host__ void 
launchDummyKernel() {
	dummyKernel<<<1, 1>>>();
}


// converts clock_t value into seconds
__host__ dTyp 
seconds(clock_t dt) {
	return ((dTyp) dt) / ((dTyp)CLOCKS_PER_SEC);
}

void scale_x(dTyp *x, int size){
	// ensures that x \in [-1/2, 1/2)

	dTyp invrange = 1./(x[size-1] - x[0]);
	for(int i = 0; i < size; i++){
		x[i]-=x[0];
		x[i]*=invrange;
		x[i]-=0.5;
	}
}

__host__ void 
init_plan( plan *p, const dTyp *x, const dTyp *f, int Ndata, int Ngrid,
           unsigned int plan_flags) {
	LOG("in init_plan -- mallocing for CPU");
	p->Ndata = Ndata;
	p->Ngrid = Ngrid;
	p->flags = plan_flags;
	//p->out_root = NULL;

	p->x_data = (dTyp *)    malloc( Ndata * sizeof(dTyp));
	p->f_data = (dTyp *)    malloc( Ndata * sizeof(dTyp));
	p->f_hat  = (Complex *) malloc( Ngrid * sizeof(Complex));
	p->f_hat_win = (Complex *) malloc(Ngrid * sizeof(Complex));

	LOG("memcpy x and f to plan");
	memcpy(p->x_data, x, Ndata * sizeof(dTyp));
	
	if (f == NULL) {
		eprint("you passed a NULL pointer to init_plan.\n");
	} else {
		memcpy(p->f_data, f, Ndata * sizeof(dTyp));
        }

	// Allocate GPU memory
	LOG("cudaMalloc -- p->g_f_data");
	checkCudaErrors(
		cudaMalloc((void **) &(p->g_f_data), 
			p->Ndata * sizeof(dTyp))
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

	if( p->flags & CALCULATE_WINDOW_FUNCTION ) {
		LOG("initializing memory for window function");
		p->f_hat_win = (Complex *) malloc( Ngrid * sizeof(Complex));
	
		// window function
		checkCudaErrors(
			cudaMalloc((void **) &(p->g_f_hat_win), 
				p->Ngrid * sizeof(Complex))
		);

		// set to zero
	    checkCudaErrors(
			cudaMemset(p->g_f_hat_win, 0, p->Ngrid * sizeof(Complex))
		);
	}

		
    // set things to zero
	checkCudaErrors(
		cudaMemset(p->g_f_hat, 0, p->Ngrid * sizeof(Complex))
	);
	

	LOG("cudaMemcpy f_data_complex ==> p->g_f_data");
	// Copy f_j -> GPU
	checkCudaErrors(
		cudaMemcpy(p->g_f_data, p->f_data, 
			p->Ndata * sizeof(dTyp), cudaMemcpyHostToDevice)
	);
	
	LOG("cudaMemcpy p->x_data ==> p->g_x_data");
	// Copy x_j -> GPU
	checkCudaErrors(
		cudaMemcpy(p->g_x_data, p->x_data, 
			p->Ndata * sizeof(dTyp), cudaMemcpyHostToDevice)
	);

	checkCudaErrors(cudaDeviceSynchronize());

	LOG("done here, calling set_filter_properties");
	// copy filter information + perform 
	// precomputation
	set_filter_properties(p);
}

__host__ void
printComplex_d(Complex *a_d, int N, FILE* out){
	Complex * cpu = (Complex *)malloc( N * sizeof(Complex));
	checkCudaErrors(cudaMemcpy(cpu, a_d, N * sizeof(Complex), cudaMemcpyDeviceToHost ));

	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e %-10.3e\n", i, cpu[i].x, cpu[i].y);
        free(cpu);
}

__host__ void
printReal_d(dTyp *a, int N, FILE *out){
	dTyp * copy = (dTyp *) malloc(N * sizeof(dTyp));
        checkCudaErrors(cudaMemcpy(copy, a, N * sizeof(dTyp), cudaMemcpyDeviceToHost));

	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e\n", i, copy[i]);
        free(copy);
}
__host__ void
printComplex(Complex *a, int N, FILE *out){
	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e %-10.3e\n",  i, a[i].x, a[i].y);
}

__host__ void
printReal(dTyp *a, int N, FILE *out){
	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e\n",  i, a[i]);
}



void print_plan(plan *p) {
    fprintf(stderr, "PLAN: \n\tp->Ngrid = %d\n\tp->Ndata = %d\n",
            p->Ngrid, p->Ndata);
    fprintf(stderr, "  plan->f_data\n");
    printReal(p->f_data, p->Ndata, stderr);
    fflush(stderr);

    fprintf(stderr, "  plan->x_data\n");
    printReal(p->x_data, p->Ndata, stderr);
    fflush(stderr);

    fprintf(stderr, "  plan->g_x_data\n");
    printReal_d(p->g_x_data, p->Ndata, stderr);

    checkCudaErrors(cudaDeviceSynchronize());

    fprintf(stderr, "  plan->g_f_data\n");
    //printReal_d(p->g_f_data, p->Ndata, stdout);

    checkCudaErrors(cudaDeviceSynchronize());

    fprintf(stderr, "  plan->g_f_hat\n");
    printComplex_d(p->g_f_hat, p->Ngrid, stderr);

    checkCudaErrors(cudaDeviceSynchronize());

    //fprintf(stderr, "  plan->g_f_filter\n");
    //printComplex_d(p->g_f_filter, p->Ngrid, stdout);

    //checkCudaErrors(cudaDeviceSynchronize());


}

void free_plan(plan *p){
	LOG("===== free_plan =====");
	LOG("free     p->f_hat");
	free(p->f_hat);
	LOG("free     p->x_data");
	free(p->x_data);
	LOG("free     p->f_data");
	free(p->f_data);

	LOG("free     p->(fprops_device, fprops_host)");
	free_filter_properties(p->fprops_device, p->fprops_host);

	LOG("cudaFree p->g_f_hat");
	checkCudaErrors(cudaFree(p->g_f_hat));

	LOG("cudaFree p->g_f_data");
	checkCudaErrors(cudaFree(p->g_f_data));

	LOG("cudaFree p->g_x_data");
	checkCudaErrors(cudaFree(p->g_x_data));


	if(p->flags & CALCULATE_WINDOW_FUNCTION) {
		checkCudaErrors(cudaFree(p->g_f_hat_win));
		free(p->f_hat_win);
	}

	LOG("free     p");
	free(p);

	LOG("=====================");
}




__global__ void print_filter_props_d(filter_properties *f, int Ndata){
	printf("DEVICE FILTER_PROPERTIES\n\tb = %.3e\n\tfilter_radius = %d\n", f->b, f->filter_radius);
	for(int i = 0; i < Ndata; i++)
		printf("\tf->E1[%-3d] = %-10.3e\n", i, f->E1[i]);
	printf("\t---------------------\n");
	for(int i = 0; i < Ndata; i++)
		printf("\tf->E2[%-3d] = %-10.3e\n", i, f->E2[i]); 
	printf("\t---------------------\n");
	for(int i = 0; i < f->filter_radius; i++)
		printf("\tf->E3[%-3d] = %-10.3e\n", i, f->E3[i]); 
	printf("\t---------------------\n");
}
