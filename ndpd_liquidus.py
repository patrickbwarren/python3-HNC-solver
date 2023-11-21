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

# Use the HNC package to solve nDPD potentials, optionally plotting
# the pair distribution function and potential.  The nDPD model is
# described in Sokhan et al., Soft Matter 19, 5824 (2023).

# Extracted from Fig 5 in this paper, at T* = T / T_c = 0.4 the
# liquidus points are ρ* = ρ / ρc = 4.15 (n = 3) and 3.56 (n = 4).
# These state points can be refined with:
# ./ndpd_liquidus.py -n 3 -T 0.4 -r 4.2,4.3 --> ρ = 4.26955
# ./ndpd_liquidus.py -n 4 -T 0.4 -r 3.5,3.6 --> ρ = 3.5734

import os
import pyHNC
import argparse
import numpy as np
import pyHNC
from numpy import pi as π
from pyHNC import truncate_to_zero, ExtendedArgumentParser

parser = ExtendedArgumentParser(description='nDPD HNC calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser, alpha=0.01, npicard=20000) # greatly reduce alpha and increase npicard here !!
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('-n', '--n', default='3', help='governing exponent, default 2')
parser.add_argument('-A', '--A', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-B', '--B', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-T', '--T', default=1.0, type=float, help='temperature, default 1.0')
parser.add_argument('-r', '--rho', default='4.2,4.3', help='bracketing density, default 4.0')
parser.add_argument('--ptol', default=1e-4, type=float, help='condition for vanishing pressure')
parser.add_argument('--np', default=10, type=int, help='max number of iterations')
parser.add_bool_arg('--relative', default=True, help='ρ, T relative to critical values')
args = parser.parse_args()

args.script = os.path.basename(__file__)

# The following are Tables from Sokhan et al., Soft Matter 19, 5824 (2023).

Table1 = {'2': (2, 25.0, 3.02), # n, A, B, values
          '3': (3, 15.0, 7.2),
          '4': (4, 10.0, 15.0)}

Table2 = {'2': (1.025, 0.2951, 0.519), # T_c, p_c, ρ_c values
          '3': (1.283, 0.3979, 0.504),
          '4': (1.290, 0.4095, 0.484)}

if args.n in Table1:
    n, A, B = Table1[args.n] # unpack the default values
    Tc, _, ρcσ3 = Table2[args.n] # where available
else:
    print(f'{args.script}: currently n is restricted to', ', '.join(Table1.keys()))
    exit(1)

A = args.A if args.A is not None else A # overwrite if necessary
B = args.B if args.B is not None else B # overwrite if necessary

σ = 1 - ((n+1)/(2*B))**(1/(n-1)) # the size is where the potential vanishes

ρc = ρcσ3 / σ**3 # back out the critical value

# density, temperature

ρ_vals = (ρc * pyHNC.as_linspace(args.rho)) if args.relative else pyHNC.as_linspace(args.rho)

if len(ρ_vals) == 1:
    ρ = ρ_vals[0]
else:
    ρ1, ρ2 = ρ_vals.tolist()

kT = (args.T * Tc) if args.relative else args.T
β = 1 / kT

grid = pyHNC.Grid(**pyHNC.grid_args(args)) # make the initial working grid

r, Δr = grid.r, grid.deltar # extract the co-ordinate array for use below

if args.verbose:
    print(f'{args.script}: {grid.details}')

# Define the nDPD potential as in Eq. (5) in Sokhan et al., assuming
# r_c = 1, and the derivative, then solve the HNC problem.  The arrays
# here are all size ng-1, same as r[:]

φ = truncate_to_zero(A*B/(n+1)*(1-r)**(n+1) - A/2*(1-r)**2, r, 1) # the nDPD potential
f = truncate_to_zero(A*B*(1-r)**n - A*(1-r), r, 1) # the force f = -dφ/dr

solver = pyHNC.PicardHNC(grid, nmonitor=500, **pyHNC.solver_args(args))

if args.verbose:
    print(f'{args.script}: {solver.details}')

# For the integrals here, see Eqs. (2.5.20) and (2.5.22) in Hansen &
# McDonald, "Theory of Simple Liquids" (3rd edition): for the (excess)
# energy density, e = 2πρ² ∫_0^∞ dr r² φ(r) g(r) and virial pressure,
# p = ρ + 2πρ²/3 ∫_0^∞ dr r³ f(r) g(r) where f(r) = −dφ/dr is the
# force.  An integration by parts shows that the mean-field
# contributions, being these with g(r) = 1, are the same.
# Here specifically the mean-field contributions are 
# 2πρ²/3 ∫_0^∞ dr r³ f(r) = ∫_0^1 dr r³ [AB(1-r)^n-A(1−r)]
# = πAρ²/30 * [120B/((n+1)(n+2)(n+3)(n+4)) - 1].

def pressure(ρ):
    for second_attempt in [False, True]:
        if second_attempt : # try again from cold start
            solver.warmed_up = False
        soln = solver.solve(β*φ, ρ, monitor=args.verbose) # solve model at β = 1/T
        if soln.converged:
            break
    else:
        exit(1) # failed to converge
    h = soln.hr
    p_mf = π*A*ρ**2/30*(120*B/((n+1)*(n+2)*(n+3)*(n+4)) - 1)
    p_xc = 2/3*π*ρ**2 * np.trapz(r**3*f*h, dx=Δr)
    p_ex = p_mf + p_xc
    p = ρ*kT + p_ex
    return p

if len(ρ_vals) == 1:

    p = pressure(ρ)

    print(f'{args.script}: model: nDPD with n = {n:d}, A = {A:g}, B = {B:g}, σ = {σ:g}, β = {β:g}')
    print(f'{args.script}: ρ/ρc, p =\t{ρ/ρc:g}\t{p:g}')

else:

    p1 = pressure(ρ1)
    p2 = pressure(ρ2)

    print(f'{args.script}: model: nDPD with n = {n:d}, A = {A:g}, B = {B:g}, σ = {σ:g}, β = {β:g}')
    print(f'{args.script}: iteration 000, ρ/ρc, p =\t\t{ρ1/ρc:g}\t\t{p1:g}')
    print(f'{args.script}: iteration  00, ρ/ρc, p =\t\t{ρ2/ρc:g}\t\t{p2:g}')

    if p1*p2 > 0.0:
        print(f'{args.script}: root not bracketed')
        exit(0)

    # Proceed to find where the pressure vanishes 

    for i in range(args.np):
        ρ = 0.5*(ρ1 + ρ2)
        p = pressure(ρ)
        print(f'{args.script}: iteration {i:3d}, ρ/ρc, p =\t{ρ1/ρc:g}\t{ρ/ρc:g}\t{ρ2/ρc:g}\t{p:g}')
        ρ1, ρ2 = (ρ, ρ2) if p*p1 > 0.0 else (ρ1, ρ)

    print(f'{args.script}: FINAL, ρ/ρc, p =\t{ρ/ρc:g}\t{p:g}')
