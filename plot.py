import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

os.system("rm -f *dat && ./cunfft_adjoint %s %s %s"%(sys.argv[1], sys.argv[2], sys.argv[3]))

freq = float(sys.argv[3])
dt_complex = np.dtype([ ('index', int), ('re', float), ('im', float) ])
dt_real = np.dtype([ ('x', float), ('y', float) ])


########################
## Unequally spaced

orig = np.loadtxt("original_unequally_spaced.dat", dtype=dt_real)

fft_raw_data = np.loadtxt("unequally_spaced_FFT_raw_f_hat.dat", dtype=dt_complex)
#fft_raw_filter = np.loadtxt("unequally_spaced_FFT_raw_f_filter.dat", dtype=dt_complex)

gridded_data = np.loadtxt('unequally_spaced_gridded_data.dat', dtype=dt_complex)
gridded_filter = np.loadtxt('unequally_spaced_gridded_filter.dat', dtype=dt_complex)

fft_adjoint = np.loadtxt('unequally_spaced_FFT.dat', dtype=dt_complex)

########################
## Equally spaced
orig_equal = np.loadtxt("original_equally_spaced.dat", dtype=dt_real)

fft_raw_data_equal = np.loadtxt("equally_spaced_FFT_raw_f_hat.dat", dtype=dt_complex)
#fft_raw_filter_equal = np.loadtxt("equally_spaced_FFT_raw_f_filter.dat", dtype=dt_complex)

#gridded_data_equal = np.loadtxt('equally_spaced_gridded_data.dat', dtype=dt_complex)
#gridded_filter_equal = np.loadtxt('equally_spaced_gridded_filter.dat', dtype=dt_complex)

fft_adjoint_equal = np.loadtxt('equally_spaced_FFT.dat', dtype=dt_complex)

########################
T = max(orig['x']) - min(orig['x'])

fig, ax = plt.subplots()

ax.plot(orig['x'], orig['y'], 'ko-', label="original")
ax.plot(gridded_data['index'] * T / len(gridded_data), gridded_data['re'], 'k:', label="gridded")
ax.plot(gridded_filter['index'] * T / len(gridded_filter), gridded_filter['re'] - 1, 'c-', label="gridded filter - 1", alpha=0.5)
#ax.plot(gridded_filter_equal['index'] * T / len(gridded_filter_equal), gridded_filter_equal['re'] - 1, 'r-', label="EQSP gridded filter - 1", alpha=0.5)
ax.legend(loc='best')

fig2, ax2 = plt.subplots()


mod = lambda a: np.sqrt(np.power(a['re'], 2) + np.power(a['im'], 2))

kmax = np.argmax(mod(fft_raw_data))
fmax = fft_raw_data['index'][kmax] * 2 * np.pi / T
print fmax / freq

kmax_should_be =  freq * T / (2 * np.pi)
print kmax, kmax_should_be

ax2.plot(fft_raw_data['index'] * 2 * np.pi / T, mod(fft_raw_data), 'k:', label="FFT(gridded data) [raw]" )
ax2.plot(fft_raw_data_equal['index'] * 2 * np.pi / T, mod(fft_raw_data_equal), 'r:', label="EQSP FFT(gridded data) [raw]" )
#ax2.plot(fft_raw_filter['index'] * 2 * np.pi / T, mod(fft_raw_filter), label="FFT(filter) [raw]" )
ax2.plot(fft_adjoint['index'] * 2 * np.pi / T, mod(fft_adjoint), color='k', lw=2, label="FFT adjoint" , alpha = 0.6)
ax2.plot(fft_adjoint_equal['index'] * 2 * np.pi / T, mod(fft_adjoint_equal), color='r', lw=2, label="EQSP FFT adjoint" , alpha = 0.6 )

ax2.set_ylim(0, 1.3*max(mod(fft_raw_data)))
ax2.legend(loc='best')
ax2.axvline(freq, ls='--', color='r')

fig.savefig("org_and_gridded_signal.png")
fig2.savefig('FFTs.png')

plt.show()
