/* main.cpp
* Copyright 2016 John Hoffman
*
*/

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

#include "nfft_adjoint.h"
#include "typedefs.h"

#define rmax 100000
#define freq 10.0
#define Random ((dTyp) (rand() % rmax))/rmax


void print_plan(plan *p) {
    fprintf(stderr, "PLAN: \n\tp->Ngrid = %d\n\tp->Ndata = %d\n",
            p->Ngrid, p->Ndata);
    fprintf(stderr, "\tp->f_hat[0] = %.3e + %.3ei\n",
            p->f_hat[0].x, p->f_hat[0].y);
}

void set_freqs( dTyp *f, dTyp L, unsigned int N){
    for(int i = 0; i < N; i++)
        f[i] = i / L;
}

dTyp mag( Complex value){
    if (value.x == 0 && value.y == 0) return 0.0;
    return sqrt(value.x * value.x + value.y * value.y);
}

int main(int argc, char *argv[]) {
    unsigned int N = 20;
    dTyp f[N], x[N];

    //cudaSetDevice(0);

    LOG("setting up data array.");
    int i;



    x[0] = 0;
    for (i = 1; i < N; i++) x[i] = Random + x[i - 1];
    for (i = 1; i < N; i++) x[i] = (x[i] / x[N - 1]) * 2 * PI;

    for (i = 0; i < N; i++) f[i] = cos(freq * x[i]);

    LOG("setting x");
    dTyp range = x[N - 1] - x[0];
    scale_x(x, N);

    LOG("done.");

    plan *p = (plan *) malloc(sizeof(plan));

    LOG("about to do init_plan.");
    init_plan(p, f, x, N, 32);
    
    //print_plan(p);

    LOG("about to do nfft adjoint.");
    cuda_nfft_adjoint(p);

    Complex *f_hat = (Complex *)malloc(p->Ngrid * sizeof(Complex));
    memcpy(f_hat, p->f_hat, p->Ngrid * sizeof(Complex));

    for (i = 0; i < p->Ngrid; i++ ) 
        printf("%-15.5e %-15.5e\n", i / range, p->f_hat[i].x);

    LOG("about to free plan.");
    free_plan(p);

    return EXIT_SUCCESS;
}
