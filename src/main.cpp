
#include "typedefs.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define rmax 100000
#define freq 10.0
#define Random ((dTyp) (rand() % rmax))/rmax

int main(int argc, char *argv[]){
	int N = 1000;
	dTyp f[N], x[N];
	
	int i;
	x[0] = 0;
	for(i=1; i < N; i++) x[i] = Random + x[i-1];
	for(i=1; i < N; i++) x[i] = (x[i] / x[i-1]) * 2 * PI;

	for(i=0; i < N; i++) f[i] = cos(freq * x[i]);

	plan p;
	p.Ndata = N;
	p.Ngrid = 5*N;

	p.x_data = (dTyp *) malloc(p.Ndata * sizeof(dTyp) );
	p.f_data = (dTyp *) malloc(p.Ngrid * sizeof(dTyp) );
	return EXIT_SUCCESS;
}
