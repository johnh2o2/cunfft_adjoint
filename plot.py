import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

dtype = np.dtype([ ('f', float), ('p', float)] )

os.system('./testing l %s %s %s > lsp.dat'%(sys.argv[1], sys.argv[2], sys.argv[3]))

freq = 120.

lsp = np.loadtxt('lsp.dat', dtype=dtype)

f, ax = plt.subplots()
ax.plot(lsp['f'], lsp['p'])
ax.axvline(freq, color='r', ls='--')

ax.set_yscale('log')
ax.set_xscale('log')

#f.savefig('lsp.png')
plt.show()