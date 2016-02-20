
#include "adjoint_kernel.cuh"

__device__ void smooth_to_grid(Complex *f_data, Complex *f_grid, const unsigned int j, const unsigned int i, filter_properties *fprops){
	dTyp val;
	for (unsigned int m = -g_fprops->filter_radius + 1; 
			  m < g_fprops->filter_radius; 
                          m++)
	{
		if (f_data == NULL) val = 1.0;
		else val = f_data[j];

		val *= filter(j, i, m, fprops);
		atomicAdd(&(f_grid[i].x), val);
	}
}

__global__ void fast_gridding(Complex *f_data, Complex *f_grid, 
		const float *x_data, const unsigned int Ngrid, 
		const unsigned int Ndata, filter_properties *fprops){
	
	unsigned int i = get_index();
	
	if (i < Ndata) {
		unsigned int j = (int) ((x[i] / (2 * PI)) * Ndata);
		smooth_to_grid(f_data, f_grid, j, i, fprops);
	}
}

__global__ divide_by_spectral_window(Complex *sig, const Complex *filt,
				const unsigned int N){
	unsigned int i; 
	if (i < N) sig[i].x = sig[i].x/filt[i].x;
}
