__global__ filter_properties *g_fprops;


__device__ void smooth_to_grid(Complex *f_grid, const unsigned int index, 
				const Complex y, const float ioffset, const float dx, const unsigned int filterRadius){
	// REWRITE
	float val = 0;
	for (unsigned int i = -filterRadius + 1; i < filterRadius; i++){
		val = y * smoothing_filter(); 
		atomicAdd(&(f_grid[i].x), val);
	}

}

__global__ void fast_gaussian_gridding(Complex *f_data, Complex *f_grid, 
					const float *x_data, const unsigned int Ngrid, 
					const unsigned int Ndata){
	int i = // thread id
	
	if (i < Ndata) {
		float findex = (Ngrid * (x_data[i] + 0.5));
		if (f_data == NULL)
			smooth_to_grid(f_grid, (int) findex, 1.0, findex - ((int) findex), 1.0/Ngrid, Ngrid / Ndata)
		else
			smooth_to_grid(f_grid, (int) findex, f_data[i], findex - ((int) findex), 1.0/Ngrid, Ngrid / Ndata)
	}
}

__global__ divide_by_spectral_window(Complex *sig, Complex *filt, size_t N){
	unsigned int i; 
	if (i < N) sig[i] = sig[i]/filt[i];
}