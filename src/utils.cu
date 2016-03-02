#include "utils.h"
#include "filter.h"
#include <stdlib.h>


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


// Copies over a float array to Complex array
// TODO: Find a more efficient/sensible way to do this.
void copy_float_to_complex(dTyp *a, Complex *b, int N){
	for (int i = 0; i < N; i++){
		b[i].x = a[i];
		b[i].y = 0;
	}
}

void scale_x(dTyp *x, int size){
	// ensures that x \in [0, 2pi)

	float range = x[size-1] - x[0];
	for(int i = 0; i < size; i++){
		x[i]-=x[0];
		x[i]/=range;
		x[i] *= 2 * PI;
	}
}

void
printComplex_d(Complex *a_d, int N, FILE* out){
	Complex cpu[N];
	checkCudaErrors(cudaMemcpy(cpu, a_d, N * sizeof(Complex), cudaMemcpyDeviceToHost ));

	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e %-10.3e\n", i, cpu[i].x, cpu[i].y);
}

__global__
void
printReal_d(dTyp *a, int N){
	for(int i = 0;i < N; i++)
		printf("%-5s%-5d %-10.3e\n", " ", i, a[i]);
}
__host__
void
printComplex(Complex *a, int N){
	for(int i = 0;i < N; i++)
		printf("%-5s%-5d %-10.3e %-10.3e\n", " ", i, a[i].x, a[i].y);
}

__host__
void
printReal(dTyp *a, int N){
	for(int i = 0;i < N; i++)
		printf("%-5s%-5d %-10.3e\n", " ", i, a[i]);
}



void print_plan(plan *p) {
    fprintf(stderr, "PLAN: \n\tp->Ngrid = %d\n\tp->Ndata = %d\n",
            p->Ngrid, p->Ndata);
    fprintf(stderr, "  plan->f_data\n");
    printReal(p->f_data, p->Ndata);
    fflush(stdout);
    fflush(stderr);

    fprintf(stderr, "  plan->x_data\n");
    printReal(p->x_data, p->Ndata);
    fflush(stdout);
    fflush(stderr);

    fprintf(stderr, "  plan->g_x_data\n");
    printReal_d<<<1, 1>>>(p->g_x_data, p->Ndata);

    checkCudaErrors(cudaDeviceSynchronize());

    fprintf(stderr, "  plan->g_f_data\n");
    printComplex_d(p->g_f_data, p->Ndata, stdout);

    checkCudaErrors(cudaDeviceSynchronize());

    fprintf(stderr, "  plan->g_f_hat\n");
    printComplex_d(p->g_f_hat, p->Ngrid, stdout);

    checkCudaErrors(cudaDeviceSynchronize());

    fprintf(stderr, "  plan->g_f_filter\n");
    printComplex_d(p->g_f_filter, p->Ngrid, stdout);

    checkCudaErrors(cudaDeviceSynchronize());


}

void free_plan(plan *p){
	LOG("===== free_plan =====");
	LOG("free     p->f_hat");
	free(p->f_hat);
	LOG("free     p->x_data");
	free(p->x_data);
	LOG("free     p->f_data");
	free(p->f_data);

	LOG("cudaFree p->fprops_host->E(1,2,3)");
	checkCudaErrors(cudaFree(p->fprops_host->E1));
	checkCudaErrors(cudaFree(p->fprops_host->E2));
	checkCudaErrors(cudaFree(p->fprops_host->E3));

	LOG("free     p->fprops_host");
	free(p->fprops_host);

	LOG("cudaFree p->fprops_device");
	checkCudaErrors(cudaFree(p->fprops_device));

	LOG("cudaFree p->g_f_hat");
	checkCudaErrors(cudaFree(p->g_f_hat));

	LOG("cudaFree p->g_f_filter");
	checkCudaErrors(cudaFree(p->g_f_filter));

	LOG("cudaFree p->g_f_data");
	checkCudaErrors(cudaFree(p->g_f_data));

	LOG("cudaFree p->g_x_data");
	checkCudaErrors(cudaFree(p->g_x_data));

	LOG("free     p");
	free(p);

	LOG("=====================");
}



__host__
void 
init_plan(
	plan 			*p, 
	dTyp 			*f, 
	dTyp 			*x, 
	int 	Ndata, 
	int 	Ngrid

){
	LOG("in init_plan -- mallocing for CPU");
	p->Ndata = Ndata;
	p->Ngrid = Ngrid;
	p->x_data = (dTyp *)    malloc( Ndata * sizeof(dTyp));
	p->f_data = (dTyp *)    malloc( Ndata * sizeof(dTyp));
	p->f_hat  = (Complex *) malloc( Ngrid * sizeof(Complex));

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

	checkCudaErrors(cudaMemset(p->g_f_hat, 0, p->Ngrid * sizeof(Complex)));
	checkCudaErrors(cudaMemset(p->g_f_filter, 0, p->Ngrid * sizeof(Complex)));

	checkCudaErrors(cudaDeviceSynchronize());

	LOG("copying f_data to f_data_complex");
	// "Cast" float array to Complex array
	Complex f_data_complex[p->Ndata];
	copy_float_to_complex(p->f_data, f_data_complex, p->Ndata);

	LOG("cudaMemcpy f_data_complex ==> p->g_f_data");
	// Copy f_j -> GPU
	checkCudaErrors(
		cudaMemcpy(p->g_f_data, f_data_complex, 
			p->Ndata * sizeof(Complex), cudaMemcpyHostToDevice)
	);

	checkCudaErrors(cudaDeviceSynchronize());

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
