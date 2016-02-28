/* main.cpp
* Copyright 2016 John Hoffman
*
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "nfft_adjoint.h"

#define rmax 100000
#define freq 10.0
#define Random ((dTyp) (rand_r() % rmax))/rmax


void print_plan(plan *p) {
    fprintf(stderr, "PLAN: \n\tp->Ngrid = %d\n\tp->Ndata = %d\n",
            p->Ngrid, p->Ndata);
}

int main(int argc, char *argv[]) {
    unsigned int N = 20;
    dTyp f[N], x[N];

    //cudaSetDevice(0);

    LOG("setting up data array.");
    int i;



    x[0] = 0;
    for (i = 1; i < N; i++) x[i] = Random + x[i - 1];
    for (i = 1; i < N; i++) x[i] = (x[i] / x[i - 1]) * 2 * PI;

    for (i = 0; i < N; i++) f[i] = cos(freq * x[i]);

    LOG("setting x");
    dTyp range = x[N - 1] - x[0];
    scale_x(x, N);

    LOG("done.");

    plan p;

    LOG("about to do init_plan.");
    init_plan(&p, f, x, N, N * 5);
    print_plan(&p);


    LOG("about to do nfft adjoint.");
    cuda_nfft_adjoint(&p);

    LOG("about to free plan.");
    free_plan(&p);

    return EXIT_SUCCESS;
}
