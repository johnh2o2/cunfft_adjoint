
#include "nfft_adjoint.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define rmax 100000
#define freq 10.0
#define Random ((dTyp) (rand() % rmax))/rmax


int main(int argc, char *argv[]){
	int N = 1000;
	dTyp f[N], x[N];
	
	LOG("setting up data array.");
	int i;
	x[0] = 0;
	for(i=1; i < N; i++) x[i] = Random + x[i-1];
	for(i=1; i < N; i++) x[i] = (x[i] / x[i-1]) * 2 * PI;

	for(i=0; i < N; i++) f[i] = cos(freq * x[i]);
	LOG("done.")

	plan p;
	
	LOG("about to do init_plan.");
	init_plan(&p, f, x, N, N * 5);

	LOG("about to do nfft adjoint.");
	cuda_nfft_adjoint(&p);

	LOG("about to free plan.");
	free_plan(&p);

	return EXIT_SUCCESS;
}
