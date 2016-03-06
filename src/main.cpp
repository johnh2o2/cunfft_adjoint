/*   main.cpp
 *   ========   
 *   
 *   UNIT TESTING for the cuNFFT_adjoint operations
 * 
 *   (c) John Hoffman 2016
 * 
 *   This file is part of cuNFFT_adjoint
 *
 *   cuNFFT_adjoint is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   cuNFFT_adjoint is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with cuNFFT_adjoint.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>

#include "nfft_adjoint.h"
#include "typedefs.h"

#define rmax 1000000
#define Random ((dTyp) (rand() % rmax))/rmax

// computes |z|^2 for z \in C
dTyp complexMod2(Complex a){
    return a.x * a.x + a.y * a.y;
}

// converts clock_t value into seconds
dTyp seconds(clock_t dt) {
	return ((dTyp) dt) / ((dTyp)CLOCKS_PER_SEC);
}

// generates unequal timing array
dTyp * generateRandomTimes(int N) {
	dTyp *x = (dTyp *) malloc( N * sizeof(dTyp));
	x[0] = 0.;
	for(int i = 1; i < N; i++) 
		x[i] = x[i-1] + Random;
	
	dTyp xmax = x[N-1];
	for(int i = 0; i < N; i++)
		x[i] = (x[i] / xmax) - 0.5;

	return x; 
}

// generates a periodic signal
dTyp * generateSignal(dTyp *x, dTyp f, dTyp phi, int N) {
	dTyp *signal = (dTyp *) malloc( N * sizeof(dTyp));
	
	for(int i = 0; i < N; i++)
		signal[i] = cos((x[i] + 0.5) * f * 2 * PI - phi) + Random;

	return signal;
}


// checks if any nans are in the fft
int countNans(Complex *fft, int N){
	int nans = 0;
	for (int i = 0 ; i < N; i++)
		if (isnan(fft[i].x) || isnan(fft[i].y))
			nans++;
	return nans;
}

// wrapper for cuda_nfft_adjoint
void nfftAdjoint(dTyp *x, dTyp *f, Complex *fft, int n, int ng) {
	plan *p = (plan *) malloc(sizeof(plan));
	
	init_plan(p, x, f, n, ng);
        p->output_intermediate = 1;
	
	cuda_nfft_adjoint(p);

	memcpy(fft, p->f_hat, ng * sizeof(Complex));

	free_plan(p);
}

// Do timing tests
void timing(int Nmin, int Nmax, int Ntests){
	int      n, ng, nnans, dN = (Nmax - Nmin) / Ntests;
	dTyp     *x, *f;
	Complex  *fft;
	plan     *p;
	clock_t  start, dt;

	for(int i = 0; i < Ntests; i++){
		n = Nmin + dN * i;

		// make Ngrid = 2^a , smallest a such that Ngrid > N
		ng = (int) pow(2, ((int) log2(n)) + 1); 
		
		// generate a signal.
		x = generateRandomTimes(n);
		f = generateSignal(x, 10., 0.5, n);
		fft = (Complex *)malloc(ng * sizeof(Complex));
		
		// initialize
		p = (plan *) malloc(sizeof(plan));
		//printf("%d, %d\n", n, ng);
		init_plan(p, x, f, n, ng);

		// calculate nfft
		start = clock(); 
		cuda_nfft_adjoint(p);
		dt = clock() - start;

		// output
		nnans = countNans(fft, ng);
		printf("%-10d %-10d %d %-10.3e\n",n, ng, nnans, seconds(dt));
		
		// free
		free(fft);
		free_plan(p);
	}
}

void simple(int N, float R, float f) { 
	dTyp *x, *y;
	int Ng = ((int) R * N);
	
	Complex *fft = (Complex *) malloc( Ng * sizeof(Complex));
	
	x = generateRandomTimes(N);
	y = generateSignal(x, f, 0., N);
	
	nfftAdjoint(x, y, fft, N, Ng);

	// OUTPUT
	FILE *out;
	
    	out = fopen("original_signal.dat", "w");
    	for (int i=0; i < N; i++)
    		fprintf(out, "%e %e\n", x[i], y[i]);
    	fclose(out);

	out = fopen("adjoint_nfft.dat", "w");
	for (int i=0; i < N; i++)
		fprintf(out, "%d %e %e\n", i, fft[i].x, fft[i].y);
	fclose(out);
}

int main(int argc, char *argv[]) {
    if(argc != 5) { 
        fprintf(stderr, "usage: (1) %s s <n> <r> <f>\n", argv[0]);
	fprintf(stderr, "       (2) %s t <nmin> <nmax> <ntests>\n\n", argv[0]);
        fprintf(stderr, "n      : number of data points\n");
        fprintf(stderr, "r      : Oversampling factor\n");
        fprintf(stderr, "f      : angular frequency of signal\n");
	fprintf(stderr, "nmin   : Smallest data size\n");
	fprintf(stderr, "nmax   : Largest data size\n");
	fprintf(stderr, "ntests : Number of runs\n");
   
        exit(EXIT_FAILURE);
    }

    if (argv[1][0] == 's')
        simple(atoi(argv[2]), atof(argv[3]), atof(argv[4]));

    else if (argv[1][0] == 't')
        timing(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    
    else {
        fprintf(stderr, "What does %c mean? Should be either 's' or 't'.\n", argv[1][0]);
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
