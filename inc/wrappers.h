#ifndef _WRAPPERS
#define _WRAPPERS

#include <complex.h>

//#ifdef __cplusplus
//extern "C" {
//#endif
// another wrapper -- this is to be used by other programs.
void cunfftAdjoint(const dTyp *x, const dTyp *f, cTyp *fft, int n, int ng);
void gpuInit();
//#ifdef __cplusplus
//}
//#endif
#endif
