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
#include <complex.h>
#include <cuda.h>

#include "nfft_adjoint.h"
#include "typedefs.h"
#include "wrappers.h"
#include "utils.h"
#include "ls.h"

#define FREQUENCY 120.0
#define PHASE_SHIFT 0.5
#define HIFAC 2 
#define OVERSAMPLING 10


void gpuInit() { launchDummyKernel(); }

// wrapper for cuda_nfft_adjoint
void nfftAdjoint(dTyp *x, dTyp *f, Complex *fft, int n, int ng) {
	plan *p = (plan *) malloc(sizeof(plan));

	init_plan(p, x, f, n, ng, OUTPUT_INTERMEDIATE);

	cuda_nfft_adjoint(p);

	memcpy(fft, p->f_hat, ng * sizeof(Complex));

	free_plan(p);
}

// another wrapper -- this is to be used by other programs.
void cunfftAdjoint(const dTyp *x, const dTyp *f, cTyp *fft, int n, int ng) {
	plan *p = (plan *) malloc(sizeof(plan));

	init_plan(p, x, f, n, ng, OUTPUT_INTERMEDIATE);

	cuda_nfft_adjoint(p);

	for (int i = 0; i < ng; i++)
		fft[i] = p->f_hat[i].x + _Complex_I * p->f_hat[i].y;

	//memcpy(fft, p->f_hat, ng * sizeof(Complex));
	free_plan(p);
}

dTyp * getFrequencies(dTyp *x, dTyp over, int n, int ng) {
	dTyp range = x[n - 1] - x[0];
	dTyp df = 1. / (over * range);
	dTyp *freqs = (dTyp *)malloc(ng * sizeof(dTyp));
	for (int i = 0; i < ng; i++)
		freqs[i] = (i + 1) * df;

	return freqs;

}

void testLombScargle(int n, dTyp over, dTyp hifac) {
	int ng;

	dTyp *t = generateRandomTimes(n);
	dTyp *y = generateSignal(t, 120., 0.5, n);
	dTyp *lsp = lombScargle(t, y, n, over, hifac, &ng, 0);
	dTyp *freqs = getFrequencies(t, over, n, ng);

	for (int i = 0; i < ng; i++)
		printf("%.4e %.4e\n", freqs[i], lsp[i]);

}

/*void readHATlc(char *filename, dTyp *x, dTyp *y) {


}*/

void timeLombScargle(int nmin, int nmax, int ntests) {
	int dN  = (nmax - nmin) / (ntests);
	int ng;
	clock_t start, dt;
	int test_no = 1;
	gpuInit();
	for(int n = nmin; n < nmax && test_no <= ntests; n+=dN, test_no++) {
		dTyp *t = generateRandomTimes(n);
		dTyp *y = generateSignal(t, FREQUENCY, PHASE_SHIFT, n);
		start = clock();
		lombScargle(t, y, n, OVERSAMPLING, HIFAC, &ng, PRINT_TIMING);
		dt = clock() - start;
		printf("TEST %-10d; Ndata = %-10d; Nfreq = %-10d; dt = %.5e s\n", test_no, n, ng, seconds(dt));

	}
}


// Do timing tests
void timing(int Nmin, int Nmax, int Ntests) {
	int      n, ng, nnans, dN = (Nmax - Nmin) / Ntests;
	dTyp     *x, *f;
	Complex  *fft;
	plan     *p;
	clock_t  start, dt;

	for (int i = 0; i < Ntests; i++) {
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
		init_plan(p, x, f, n, ng, 0);

		// calculate nfft
		start = clock();
		cuda_nfft_adjoint(p);
		dt = clock() - start;

		// output
		nnans = countNans(fft, ng);
		printf("%-10d %-10d %d %-10.3e\n", n, ng, nnans, seconds(dt));

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
	for (int i = 0; i < N; i++)
		fprintf(out, "%e %e\n", x[i], y[i]);
	fclose(out);

	out = fopen("adjoint_nfft.dat", "w");
	for (int i = 0; i < N; i++)
		fprintf(out, "%d %e %e\n", i, fft[i].x, fft[i].y);
	fclose(out);
}

int main(int argc, char *argv[]) {
	if (argc != 5) {
		fprintf(stderr, "usage: [simple fft test]  (1) %s s <n>    <r>    <f>\n", argv[0]);
		fprintf(stderr, "       [timing test    ]  (2) %s t <nmin> <nmax> <ntests>\n", argv[0]);
		fprintf(stderr, "       [lomb scargle   ]  (3) %s l <n>    <over> <hifac>\n", argv[0]);
		fprintf(stderr, "       [lomb sc. timing]  (4) %s L <nmin> <nmax> <ntests>\n\n", argv[0]);
		fprintf(stderr, "n      : number of data points\n");
		fprintf(stderr, "r      : Oversampling factor\n");
		fprintf(stderr, "f      : angular frequency of signal\n");
		fprintf(stderr, "nmin   : Smallest data size\n");
		fprintf(stderr, "nmax   : Largest data size\n");
		fprintf(stderr, "ntests : Number of runs\n");
		fprintf(stderr, "over   : oversampling factor\n");
		fprintf(stderr, "hifac  : high frequency factor\n");
		exit(EXIT_FAILURE);
	}

	if (argv[1][0] == 's')
		simple(atoi(argv[2]), atof(argv[3]), atof(argv[4]));

	else if (argv[1][0] == 't')
		timing(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));

	else if (argv[1][0] == 'l')
		testLombScargle(atoi(argv[2]), atof(argv[3]), atof(argv[4]));

	else if (argv[1][0] == 'L')
		timeLombScargle(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
	else {
		fprintf(stderr, "What does %c mean? Should be either 's', 't', or 'l'.\n", argv[1][0]);
		exit(EXIT_FAILURE);
	}

	return EXIT_SUCCESS;
}

