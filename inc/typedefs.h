
#ifndef TYPEDEFS_
#define TYPEDEFS_

#include <vector_types.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// #define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
#define dTyp double
#define Complex doubleComplex
#define fftComplex cufftDoubleComplex
#else
#define dTyp float
#define Complex singleComplex
#define fftComplex cufftComplex
#endif


#define PI 3.14159265358979323846

#define eprint(...) \
    fprintf(stderr, "[%-10s] %-30s L[%-5d]: ", "ERROR", __FILE__, __LINE__);\
    fprintf(stderr, __VA_ARGS__);

//#define LOG(msg) 
#ifdef DEBUG
    #define LOG(msg) fprintf(stderr, "[%-10s] %-30s L[%-5d]: %s\n", "OK", __FILE__, __LINE__, msg)
#else
    #define LOG(msg) 
#endif
#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess)
    {
        fprintf(stderr, "ERROR %-24s L[%-5d]: %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

typedef float2 singleComplex;
typedef double2 doubleComplex;

extern int EQUALLY_SPACED;

typedef enum {
    CPU_FREE,
    CUDA_FREE
} free_type;

typedef struct {
    dTyp tau;
    int filter_radius;
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

    int Ndata, Ngrid, filter_radius;

    filter_properties *fprops_host, *fprops_device;

    char out_root[100];

} plan;

void free_filter_properties(filter_properties *f, free_type how_to_free);
void free_plan(plan *p);

#endif
