
#ifndef TYPEDEFS_
#define TYPEDEFS_

typedef struct {
	float tau;
	int filter_radius;
	float *E1;
	float *E2;
	float *E3;
} filter_properties;

typedef float2 Complex;

typedef struct {
	float *x_data, *f_data;
	Complex *f_hat;

	// GPU variables
	Complex *g_f_hat, *g_f_filter, *g_f_data; 
	float *g_x_data;

	int Ndata, Ngrid, filter_radius;
} plan;

#endif