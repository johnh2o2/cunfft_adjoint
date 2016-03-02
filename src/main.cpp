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


dTyp complexMod2(Complex a){
    return a.x * a.x + a.y * a.y;
}

void set_freqs( dTyp *f, dTyp L, int N){
    for(int i = 0; i < N; i++)
        f[i] = i / L;
}

dTyp mag( Complex value){
    if (value.x == 0 && value.y == 0) return 0.0;
    return sqrt(value.x * value.x + value.y * value.y);
}

int main(int argc, char *argv[]) {
    int N = 1024;
    dTyp f[N], x[N];

    //cudaSetDevice(0);

    LOG("setting up data array.");
    int i;



    x[0] = 0;
    dTyp phi = PI/2;
    for (i = 1; i < N; i++) x[i] = Random + x[i - 1];
    for (i = 1; i < N; i++) x[i] = (x[i] / x[N - 1]) * 2 * PI;

    for (i = 0; i < N; i++) f[i] = cos(freq * x[i] - phi);



    LOG("setting x");
    dTyp range = x[N - 1] - x[0];
    scale_x(x, N);

    FILE *out = fopen("original.dat", "w");
    for (i=0; i < N; i++)
        fprintf(out, "%e %e\n", x[i] * range, f[i]);
    fclose(out);

    LOG("done.");

    plan *p = (plan *) malloc(sizeof(plan));

    LOG("about to do init_plan.");
    init_plan(p, f, x, N, 4*N);
    
    //print_plan(p);
    //return EXIT_SUCCESS;

    LOG("about to do nfft adjoint.");
    cuda_nfft_adjoint(p);
    //print_plan(p);

    Complex *f_hat = (Complex *)malloc(p->Ngrid * sizeof(Complex));
    memcpy(f_hat, p->f_hat, p->Ngrid * sizeof(Complex));

    FILE *out_result = fopen("FFT.dat", "w");
    for (i = 0; i < p->Ngrid; i++ ) 
        fprintf(out_result,"%-5d %-15.5e %-15.5e\n",i , p->f_hat[i].x, p->f_hat[i].y);
    fclose(out_result);

    LOG("about to free plan.");
    free_plan(p);

    return EXIT_SUCCESS;
}
