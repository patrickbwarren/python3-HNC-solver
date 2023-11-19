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

# This version uses the 'vanilla' DFT which results in
# −c(q) = φ(q) + 2 u'(ρbar) w(q) + u''(ρbar) [w(q)]², which closes
# the Ornstein-Zernike relation h(q) = c(q) / [1 − ρ c(q)].
# The optional 'EXP' improvement in real space is h <-- (exp(h) − 1)
# This is then used to calculate the virial pressure as
# p = ρ + 2πρ²/3 ∫_0^∞ dr r³ f(r) g(r) where g(r) = 1 + h(r) and
# the force law f(r) = − φ'(r) − 2u'(ρbar) w'(r).
# The mean-field pressure is the same evaluated with g(r) = 1, and is
# p_MF = ρ + πρ²(A + 2B(R²)²ρbar)/30.  Hence p − p_MF is the above
# integral evaluated with h(r) instead of g(r).

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
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('-A', '--A', default=10.0, type=float, help='repulsion amplitude')
parser.add_argument('-B', '--B', default=5.0, type=float, help='repulsion amplitude')
parser.add_argument('-R', '--R', default=0.75, type=float, help='repulsion r_c')
parser.add_argument('-r', '--rho', default='3.0', help='density or density range, default 3.0')
parser.add_bool_arg('--exp', default=False, help='use the EXP approximation')
parser.add_argument('--rmax', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_argument('-s', '--show', action='store_true', help='show results')
args = parser.parse_args()

args.script = os.path.basename(__file__)

A, B, R = args.A, args.B, args.R

grid = pyHNC.Grid(**pyHNC.grid_args(args)) # make the initial working grid
r, Δr, q, Δq = grid.r, grid.deltar, grid.q, grid.deltaq # extract for use below
rbyR, qR = r/R, q*R # some reduced variables

# DPD potential, force law without amplitude, and Fourier transform.
# The array sizes here are ng-1, same as r[:].

φ = A/2 * truncate_to_zero((1-r)**2, r, 1)
φf = truncate_to_zero((1-r), r, 1)
φq = 4*π*A*(2*q + q*cos(q) - 3*sin(q)) / q**5

# The many-body weight function (normalised) and its Fourier
# transform, and the derivative (unnormalised).

w = 15/(2*π*R**3) * truncate_to_zero((1-rbyR)**2, r, R)
wq = 60*(2*qR + qR*cos(qR) - 3*sin(qR)) / qR**5
wf = truncate_to_zero((1-rbyR), r, R) # omit the normalisation

K = π*B*R**4/30 # used in many expressions below

for ρ in pyHNC.as_linspace(args.rho):
    ρbar = ρ # under this approximation
    negcq = φq + 4*K*ρbar*wq + 2*ρ*K*wq**2 # note ρ in second
    hq = - negcq / (1 + ρ*negcq) # Ornstein-Zernike relation
    h = grid.fourier_bessel_backward(hq)
    h = (exp(h) - 1) if args.exp else h
    Δfg = (A*φf + 2*B*ρbar*wf) * h # features in the virial pressure integral
    pMF = ρ + π*A*ρ**2/30 + π*B*R**4*ρ**2*ρbar/15 # mean field pressure
    p = pMF + 2/3*π*ρ**2*np.trapz(r**3*Δfg, dx=Δr) # separated off MF contribution here
    print(f'{args.script}: A, B, R, ρ = {A}, {B}, {R}, {ρ}', 'pMF, p = %f\t%f' % (pMF, p))

if args.show:

    import matplotlib.pyplot as plt

    g = 1.0 + h # the pair function
    v = φ + π*B*R**4*ρbar/15 * w # generalised MB DPD potential

    cut = r < args.rmax
    plt.plot(r[cut], g[cut], 'g')        
    plt.plot(r[cut], v[cut]/10, 'r')
    plt.plot(r[cut], w[cut]/(15/(2*π*R**3)), 'b')
    plt.xlabel('$r$')
    
    plt.show()
