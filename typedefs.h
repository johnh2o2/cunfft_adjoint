
#ifndef TYPEDEFS_
#define TYPEDEFS_

#define dTyp float


typedef struct {
	dTyp tau;
	unsigned int filter_radius;
	dTyp *E1;
	dTyp *E2;
	dTyp *E3;
} filter_properties;

typedef float2 Complex;

typedef struct {
	dTyp *x_data, *f_data;
	Complex *f_hat;

	// GPU variables
	Complex *g_f_hat, *g_f_filter, *g_f_data; 
	dTyp *g_x_data;

	unsigned int Ndata, Ngrid, filter_radius;
} plan;

#endif
