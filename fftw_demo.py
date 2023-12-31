#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This program is part of pyHNC, copyright (c) 2023 Patrick B Warren (STFC).
# Email: patrick.warren{at}stfc.ac.uk.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

# Tests python implementation to FFTW.

# We want to calculate w(q) = 4π/q ∫_0^∞ dr sin(qr) r w(r),
# where w(r) = 1/4 (1 - r)² for r < 1, w(r) = 0 otherwise.
# The exact result is w(q) = 2π/q⁵ [2q + q cos(q) - 3 sin(q)].
# Also lim_(q->0) w(q) = π/30.

# From the FFTW documentation, RODFT00 implements
# Y_k = 2 ∑_(j=0)^(n-1) X_j sin(π(j+1)(k+1)/(n+1)),
# and works with arrays X_j and Y_k of length n.
# See www.fftw.org/fftw3_doc/What-FFTW-Really-Computes.html
# under "1d Real-odd DFTs (DSTs)".

# To cast this into the right form, set Δr×Δq = π/(n+1) and
# assign r_j = (j+1)×Δr for j=0 to n-1, and likewise
# q_k = (k+1)×Δq for j=0 to n-1, so that
# Y_k = 2 ∑_(j=0)^(n-1) X_j sin(r_j q_k)
# For the desired integral we can then write
# w(q_k) = 2πΔr/q_k × 2 ∑_(j=0)^(n-1) r_j w(r_j) sin(r_j q_k)
# It is this which is implemented below.
# The Fourier back transform w(r) = 1/(2π²r) ∫_0^∞ dq sin(qr) q w(q)
# is handled similarly.

# Note that the FFTW array length n which enters these expressions is
# ng-1 in the code. Experimenting, the fastest times are when the FFTW
# array sizes are n = 2^r-1 with r being an integer.  Hence, we should
# use ng = 2^r.

import pyfftw
import argparse
import numpy as np
from numpy import pi as π
from numpy import log, sin, cos

parser = argparse.ArgumentParser(description='RPM one off calculator')
parser.add_argument('--grid', default=None, help='grid definition as deltar/ng, eg 0.02/8192')
parser.add_argument('-n', '--ngrid', default='2^15', help='number of grid points, default 2^15 = 32768')
parser.add_argument('-i', '--iters', default=10, type=int, help='number of iterations for timing, default 10')
parser.add_argument('-d', '--deltar', default=0.01, type=float, help='grid spacing, default 0.01')
parser.add_argument('--rmax', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_argument('--qmax', default=15.0, type=float, help='maximum in q for plotting, default 15.0')
parser.add_argument('-s', '--show', action='store_true', help='show results')
args = parser.parse_args()

if args.grid:
    if '/' in args.grid:
        args_deltar, args.ngrid = args.grid.split('/')
        args.deltar = float(args_deltar)
    else:
        args.deltar = float(args.grid)
        r = int(1+log(π/(args.deltar**2))/log(2.0))
        args.ngrid = str(2**r)

ng = eval(args.ngrid.replace('^', '**')) # catch 2^10 etc
Δr = args.deltar
Δq = π / (Δr*ng) # equivalent to π / (Δr*(fftw_n+1))
print('ng, Δr, Δq, iters =', ng, f'(2^{int(0.5+log(ng)/log(2.0))})', Δr, Δq, args.iters)

r = Δr * np.arange(1, ng) # start from 1, and of length ng-1
q = Δq * np.arange(1, ng) # ditto
fftw_n = len(r)
print('FFTW array sizes =', fftw_n)

wr_exact = 1/4*(1-r)**2
wr_exact[r>1.0] = 0.0
wq_exact = 2*π*(2*q + q*cos(q) - 3*sin(q)) / q**5 

fftwx = pyfftw.empty_aligned(fftw_n)
fftwy = pyfftw.empty_aligned(fftw_n)
fftw = pyfftw.FFTW(fftwx, fftwy, direction='FFTW_RODFT00', flags=('FFTW_ESTIMATE',))

for i in range(args.iters):
    fftwx[:] = r * wr_exact # note the assigment here
    fftw.execute() # do the transform
    wq = 2*π*Δr/q * fftwy # unwrap the result
    fftwx[:] = q * wq_exact # the same for the back transform
    fftw.execute()
    wr = Δq/(4*π**2*r) * fftwy

if args.show:

    import matplotlib.pyplot as plt

    plt.figure(1)
    cut = r < args.rmax
    plt.plot(r[cut], wr[cut], '.')
    plt.plot(r[cut], wr_exact[cut], '--')

    plt.figure(2)
    cut = q < args.qmax
    plt.plot(q[cut], wq[cut], '.')
    plt.plot(q[cut], wq_exact[cut], '--')

    plt.show()

else:

    print('r, wr*4, wr_wexact*4 =',
          r[0], 4*wr[0], 4*wr_exact[0])

    print('q, wq*π/30, wq_wexact*π/30 =',
          q[0], 30/π*wq[0], 30/π*wq_exact[0])
