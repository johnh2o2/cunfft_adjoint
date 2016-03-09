#CUDA implementation of the NFFT adjoint operation

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
You'll need to either symlink/copy the shared libraries into the "bin"
directory, or add the "lib" directory to your `LD_LIBRARY_PATH`.

There will eventually be a `make install` and `make uninstall` option.

Two libraries are produced: `libcunaf.so` and `libcunad.so`; these 
are identical libraries except `libcunaf.so` uses **single** precision
throughout and `libcunad.so` uses **double** precision throughout.

The testing binaries allow you to do simple transforms and time
the results. To see how to use them, simply run them without any 
command line arguments:

`./test-single`

##TODO

* **Batched operations** -- if you have several transforms that you would
  like to do, it is far faster to perform them simultaneously on the GPU
  (depending on the kind of GPU you might have) to utilize all of the
  computational power.

* **More documentation** 

* **Optimizations**