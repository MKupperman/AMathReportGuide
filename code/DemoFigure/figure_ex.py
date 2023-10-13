"""
figure_ex.py
Michael Kupperman

This file generates the figure for the paper. It generates two sets of data
and plots them on a log-log scale. It then computes the slope of the data
using linear regression and prints the results.

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


np.random.seed(2023)


# Generate some sample data showing noise.
# Data should be log-linearly scaled with a slope of -1
n=20
x = np.linspace(0, n, n+1)
# We want y = 10^-(x + noise)
y1 = np.exp(-x + np.random.randn(len(x)) / 10)
y2 = np.exp( -3 -x + np.random.randn(len(x)) / 10)

plt.figure(figsize=(8, 4))
plt.semilogy(x, y1, 'o', label='Algorithm A')
plt.semilogy(x, y2, 'o', label='Algorithm B')
plt.xlabel('Number of iterations')
plt.ylabel(r'$\ell_2$ Error')
plt.title('Error of numercial solvers on Gaussian i.i.d. matrix inversion')
plt.legend()
plt.tight_layout()
plt.savefig('figure.eps')

lm1 = linregress(x, np.log(y1))
lm2 = linregress(x, np.log(y2))
print('Slope of Algorithm A: {:.6f}'.format(lm1.slope))
print('Slope of Algorithm B: {:.6f}'.format(lm2.slope))