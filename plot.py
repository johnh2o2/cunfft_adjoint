import numpy as np
import matplotlib.pyplot as plt


dt_complex = np.dtype([ ('index', int), ('re', float), ('im', float) ])
dt_real = np.dtype([ ('x', float), ('y', float) ])

orig = np.loadtxt("original.dat", dtype=dt_real)

fft_raw_data = np.loadtxt("FFT_raw_f_hat.dat", dtype=dt_complex)
#fft_raw_filter = np.loadtxt("FFT_raw_f_filter.dat", dtype=dt_complex)

gridded_data = np.loadtxt('gridded_data.dat', dtype=dt_complex)
gridded_filter = np.loadtxt('gridded_filter.dat', dtype=dt_complex)

fft_adjoint = np.loadtxt('FFT.dat', dtype=dt_complex)

T = max(orig['x']) - min(orig['x'])

fig, ax = plt.subplots()

ax.plot(orig['x'], orig['y'], 'ko-', label="original")
ax.plot(gridded_data['index'] * T / len(gridded_data), gridded_data['re'], 'k:', label="gridded")
ax.plot(gridded_filter['index'] * T / len(gridded_filter), gridded_filter['re'], 'c-', label="gridded filter")
ax.legend(loc='best')

fig2, ax2 = plt.subplots()


mod = lambda a: np.sqrt(np.power(a['re'], 2) + np.power(a['im'], 2))

ax2.plot(fft_raw_data['index'] * 2 * np.pi / T, mod(fft_raw_data), label="FFT(gridded data) [raw]" )
#ax2.plot(fft_raw_filter['index'] * 2 * np.pi / T, mod(fft_raw_filter), label="FFT(filter) [raw]" )
ax2.plot(fft_adjoint['index'] * 2 * np.pi / T, mod(fft_adjoint), color='k', lw=2, label="FFT adjoint" )
ax2.set_ylim(0, 1.3*max(mod(fft_raw_data)))
ax2.legend(loc='best')
ax2.axvline(10.0, ls='--', color='r')
plt.show()
