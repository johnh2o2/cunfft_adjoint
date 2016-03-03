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

int EQUALLY_SPACED;

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
    dTyp f[N], x[N], x_aligned[N], f_aligned[N];
    EQUALLY_SPACED = 0;
    //cudaSetDevice(0);

    LOG("setting up data array.");
    int i; char fname[200];
    dTyp range, dx;
    plan *p;
    FILE *out, *out_result;

    x[0] = 0;
    dTyp phi = 0;//PI/2;


    for (i = 1; i < N; i++) x[i] = Random + x[i - 1];
    for (i = 1; i < N; i++) x[i] = (x[i] / x[N - 1]) * 2 * PI;
    for (i = 0; i < N; i++) f[i] = cos(freq * x[i] - phi);

    range = x[N - 1] - x[0];
    dx = range/(N - 1);
    for(i=0; i < N; i++) {
        x_aligned[i] = x[0] + dx * i;
        f_aligned[i] = cos(freq * x_aligned[i] - phi);
    }

    LOG("scaling x");
    scale_x(x, N);
    scale_x(x_aligned, N);

    out = fopen("original_unequally_spaced.dat", "w");
    for (i=0; i < N; i++)
        fprintf(out, "%e %e\n", x[i] * range, f[i]);
    fclose(out);

    out = fopen("original_equally_spaced.dat", "w");
    for (i=0; i < N; i++)
        fprintf(out, "%e %e\n", x_aligned[i] * range, f_aligned[i]);
    fclose(out);
    LOG("done.");

    ///////////////////////////////////
    // UNEQUALLY SPACED
    p = (plan *) malloc(sizeof(plan));

    LOG("about to do init_plan.");
    init_plan(p, f, x, N, 4*N);
    sprintf(p->out_root, "unequally_spaced");

    LOG("about to do nfft adjoint.");
    cuda_nfft_adjoint(p);
    //print_plan(p);
    
    sprintf(fname, "%s_FFT.dat", p->out_root);
    out_result = fopen(fname, "w");
    for (i = 0; i < p->Ngrid; i++ ) 
        fprintf(out_result,"%-5d %-15.5e %-15.5e\n",i , p->f_hat[i].x, p->f_hat[i].y);
    fclose(out_result);

    LOG("about to free plan.");
    free_plan(p);

    ///////////////////////////////////
    // EQUALLY SPACED
    p = (plan *) malloc(sizeof(plan));

    LOG("about to do init_plan.");
    EQUALLY_SPACED = 1;
    init_plan(p, f_aligned, x_aligned, N, N);
    sprintf(p->out_root, "equally_spaced");

    LOG("about to do nfft adjoint.");
    cuda_nfft_adjoint(p);
    //print_plan(p);
    
    sprintf(fname, "%s_FFT.dat", p->out_root);
    out_result = fopen(fname, "w");
    for (i = 0; i < p->Ngrid; i++ ) 
        fprintf(out_result,"%-5d %-15.5e %-15.5e\n",i , p->f_hat[i].x, p->f_hat[i].y);
    fclose(out_result);

    LOG("about to free plan.");
    free_plan(p);


    return EXIT_SUCCESS;
}
