
#ifndef TYPEDEFS_
#define TYPEDEFS_

#include <vector_types.h>

#define dTyp float
#define PI 3.1415926535897932384626433832795028841971

enum {
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

typedef float2 Complex;

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
