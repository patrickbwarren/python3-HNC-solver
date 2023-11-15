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

# Generate EoS data for many-body DPD defined in PRE 68, 066702 (2003).
# In this model φ(r) = A(1−r)²/2 for r < 1 ; u(ρ) = πB(R²)²ρ²/30 ,
# and the weight function w(r) = 15/(2πR³) (1−r/R)² for r < R.

# This version uses the Percus-like DFT which results in a complicated
# route to the pair distribution function.

# This Percus-like DFT solves :
# ρ = ρ∞ exp[consts − φ ⊗ ρ − u(ρbar) − [ρ u(ρbar)] ⊗ w − U ]
# where U(r) = ( a[ρbar(r=0)] + a[ρbar] ) w(r) is the external
# potential and the consts cancel the r --> ∞ contribution from the
# remainder in the exponential.  Here '⊗' denotes a convolution.  We
# solve it in ρ(r) = ρ∞ (1 + h) where h(r) is the total correlation
# function.

import os
import sys
import pyHNC
import argparse
import subprocess
import numpy as np

from numpy import pi as π
from numpy import cos, sin, exp
from pyHNC import truncate_to_zero, ExtendedArgumentParser

parser = ExtendedArgumentParser(description='DPD EoS calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser, alpha=0.02, npicard=5000, tol=1e-12) # repurpose these here
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('-A', '--A', default=5.0, type=float, help='repulsion amplitude')
parser.add_argument('-B', '--B', default=5.0, type=float, help='repulsion amplitude')
parser.add_argument('-R', '--R', default=0.75, type=float, help='repulsion r_c')
parser.add_argument('--rho', default='3.0', help='density or density range, default 3.0')
parser.add_argument('--rmax', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_bool_arg('--show', default=False, help='show plots of things')
args = parser.parse_args()

args.script = os.path.basename(__file__)

A, B, R = args.A, args.B, args.R
α = args.alpha

grid = pyHNC.Grid(**pyHNC.grid_args(args)) # make the initial working grid
r, Δr, q, Δq = grid.r, grid.deltar, grid.q, grid.deltaq # extract for use below
rbyR, qR = r/R, q*R # some reduced variables

# DPD potential, force law, and Fourier transform.
# The array sizes here are ng-1, same as r[:].

φ = A/2 * truncate_to_zero((1-r)**2, r, 1)
φq = 4*π*A*(2*q + q*cos(q) - 3*sin(q)) / q**5
φf = A * truncate_to_zero((1-r), r, 1)

# The many-body weight function (normalised) and its Fourier
# transform, and the derivative (unnormalised).

w = 15/(2*π*R**3) * truncate_to_zero((1-rbyR)**2, r, R)
wq = 60*(2*qR + qR*cos(qR) - 3*sin(qR)) / qR**5
wf = truncate_to_zero((1-rbyR), r, R) # omit the normalisation

Bfac = π*B*R**4/30 # used in many expressions below

# for ρ in pyHNC.as_linspace(args.rho):

ρ = pyHNC.as_linspace(args.rho)[0]

h = np.zeros_like(r) # ... zero initially

v = φ + Bfac * 2*ρ * w

h = exp(-v) - 1 # initially

# Iterate from here ..

α, niter, tol = 0.02, 5000, 1e-12

for i in range(args.npicard):

    hq = grid.fourier_bessel_forward(h) # to reciprocal space
    ħq = hq * wq # convolution Δρbar = Δρ ⊗ w = ρ∞ h ⊗ w = 
    ħ = grid.fourier_bessel_backward(ħq)

    φ_term = ρ * grid.fourier_bessel_backward(φq*hq) # the φ term in the DFT, being φ ⊗ ρ

    u_term = Bfac * ρ**2 * ħ*(ħ + 2) # the u term in the DFT

    ρduρbar = 2 * Bfac * ρ**2 * (h + ħ + h*ħ) # contributing factor ρ u'(ρbar)
    ρduρbarq = grid.fourier_bessel_forward(ρduρbar) # in reciprocal space
    du_term = grid.fourier_bessel_backward(ρduρbarq*wq) # and back, as [ρ u'(ρbar)] ⊗ w

    ħ_zero = 1/(2*π**2) * np.trapz(q**2*ħq, dx=Δq) # ħ(r=0) from backward transform
    U = φ + Bfac * ρ * (2 + ħ_zero + ħ) * w # external potential in Percus DFT

    all_terms = φ_term + u_term + du_term + U
    h_new = exp(-all_terms) - 1 # the new estimate
    h = α * h_new + (1-α) * h
    error = np.sqrt(np.trapz((h_new - h)**2, dx=Δr))
    converged = error < tol
    if i % 100 == 0 or converged:
        print(f'{args.script}: iteration {i:4d}, error = {error:0.3e}')
    if converged:
        break

import matplotlib.pyplot as plt

v = φ + Bfac * 2*ρ * w


cut = r < args.rmax
plt.plot(r[cut], 1+h[cut], 'k--')
plt.plot(r[cut], ħ[cut], 'k-.')
plt.plot(r[cut], φ_term[cut], 'r')
plt.plot(r[cut], u_term[cut], 'g-.')
plt.plot(r[cut], du_term[cut], 'g--')
#plt.plot(r[cut], all_terms[cut], 'b--')
plt.plot(r[cut], U[cut]/10, 'b')
#plt.plot(r[cut], h_new[cut], 'k:')
plt.plot(r[cut], v[cut]/10, 'b:')


plt.xlabel('$r$')
    
plt.show()
