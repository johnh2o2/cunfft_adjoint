
#ifndef TYPEDEFS_
#define TYPEDEFS_

#include <vector_types.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define dTyp float
#define PI 3.1415926535897932384626433832795028841971

#define eprint(...) \
    fprintf(stderr, "ERROR %-30s L[%-5d]: ", __FILE__, __LINE__);\
    fprintf(stderr, __VA_ARGS__);

#define LOG(msg) fprintf(stderr, "%-30s L[%-5d]: %s\n", __FILE__, __LINE__, msg)

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess)
    {
        fprintf(stderr, "ERROR %-24s L[%-5d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

typedef float2 Complex;

typedef enum {
    CPU_FREE,
    CUDA_FREE
} free_type;

typedef struct {
    dTyp tau;
    unsigned int filter_radius;
    dTyp *E1;
    dTyp *E2;
    dTyp *E3;
} filter_properties;

typedef struct {
    // CPU variables
    dTyp *x_data, *f_data;
    Complex *f_hat;

    // GPU variables
    Complex *g_f_hat, *g_f_filter, *g_f_data;
    dTyp *g_x_data;

    unsigned int Ndata, Ngrid, filter_radius;

    filter_properties *fprops;
} plan;

void free_filter_properties(filter_properties *f, free_type how_to_free);
void free_plan(plan *p);

#endif
