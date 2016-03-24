#CUNA v1.2
##CUDA implementation of the NFFT adjoint operation

####(c) John Hoffman, 2016
####jah5@princeton.edu

Used for e.g. Lomb-Scargle periodograms, the adjoint operation is much
like the familiar FFT, but can be applied to unequally spaced data 

First, unevenly sampled data is smoothed onto an evenly-sampled grid
with a filter (only the Gaussian filter is available for now). Fast
Fourier transforms are then performed on the gridded data, and normalized
by the filter's window function.

##Usage

Simply `make` to produce the shared libraries and testing binaries.

Alternatively, you may run `make install` to move the libraries to
`/usr/local/lib` and headers to `/usr/local/include`.

Two libraries are produced: `libcunaf.so` and `libcunad.so`; these 
are identical libraries except `libcunaf.so` uses **single** precision
throughout and `libcunad.so` uses **double** precision throughout.

The testing binaries allow you to do simple transforms and time
the results. To see how to use them, simply run them without any 
command line arguments:

`./test-single`

##TODO

* **More documentation** 

* **Optimizations**
