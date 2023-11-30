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

import os
import pyHNC
import argparse
import numpy as np
from numpy import exp, log
from numpy import pi as π
from pyHNC import Grid, PicardHNC, truncate_to_zero, ExtendedArgumentParser

parser = ExtendedArgumentParser(description='nDPD RPA and EXP calculator')
pyHNC.add_grid_args(parser)
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('-n', '--n', default='2', help='governing exponent, default 2')
parser.add_argument('-A', '--A', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-B', '--B', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-T', '--T', default='1.0', help='temperature, default 1.0')
parser.add_argument('-r', '--rho', default='3.0', help='density or density range, default 3.0')
parser.add_argument('--rcut', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_bool_arg('--relative', default=True, help='ρ, T relative to critical values')
parser.add_bool_arg('--exp', default=False, help='use the EXP refinement of RPA')
parser.add_argument('-s', '--show', action='store_true', help='show results')
parser.add_argument('-o', '--output', help='write equation of state data to a file')
args = parser.parse_args()

args.script = os.path.basename(__file__)

# The following are Tables from Sokhan et al., Soft Matter 19, 5824 (2023).

Table1 = {'2': (2, 25.0,  3.02), # n, A, B, values
          '3': (3, 15.0,  7.2 ),
          '4': (4, 10.0, 15.0 )}

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

description = f'nDPD with n = {n:d}, A = {A:g}, B = {B:g}, σ = {σ:g}'

ρc = ρcσ3 / σ**3 # back out the critical value

# density, temperature, relative temperature

ρ_vals = (pyHNC.as_linspace(args.rho) * ρc) if args.relative else pyHNC.as_linspace(args.rho)
T = (eval(args.T.lstrip('0')) * Tc) if args.relative else eval(args.T.lstrip('0')) # allow for math in argument
β = 1 / T

grid = Grid(**pyHNC.grid_args(args)) # make a working grid

r, Δr = grid.r, grid.deltar # extract the co-ordinate array for use below

if args.verbose:
    print(f'{args.script}: {grid.details}')

# Define the nDPD potential and derivative as in Eq. (5) in Sokhan et
# al., assuming r_c = 1.  The arrays here are size ng-1, same as r[:].

φ = truncate_to_zero(A*B/(n+1)*(1-r)**(n+1) - A/2*(1-r)**2, r, 1) # the nDPD potential
f = truncate_to_zero(A*B*(1-r)**n - A*(1-r), r, 1) # the force f = -dφ/dr

if args.show:
    import matplotlib.pyplot as plt
    
# The excess virial pressure p = ρ + 2πρ²/3 ∫_0^∞ dr r³ f(r) h(r),
# where f(r) = −dφ/dr is the force: see Eq. (2.5.22) in Hansen &
# McDonald, "Theory of Simple Liquids" (3rd edition).

# The mean-field contribution is 2πρ²/3 ∫_0^∞ dr r³ f(r) = ∫_0^1 dr r³
# [AB(1−r)^n − A(1−r)] = πAρ²/30 * [120B/((n+1)(n+2)(n+3)(n+4)) − 1].

results = []

for ρ in ρ_vals:

    c = -β*φ # the RPA
    cq = grid.fourier_bessel_forward(c) # forward transform to reciprocal space
    hq = cq / (1 - ρ*cq) # solve the OZ relation
    h = grid.fourier_bessel_backward(hq) # back transform to real space
    dh = grid.fourier_bessel_backward(hq**2) # ∂h/∂ρ = h² in reciprocal space
    h = (exp(h) - 1) if args.exp else h # implement EXP if requested

    p_mf = π*A*ρ**2/30*(120*B/((n+1)*(n+2)*(n+3)*(n+4)) - 1)
    dp_mf_by_rho = π*A/15*(120*B/((n+1)*(n+2)*(n+3)*(n+4)) - 1)
    trap = np.trapz(r**3*f*h, dx=Δr)
    p_xc =  2/3*π*ρ**2 * trap
    dp_xc_by_rho = 4/3*π * trap + 2/3*π*ρ * np.trapz(r**3*f*dh, dx=Δr)
    p_ex = p_mf + p_xc
    dp_ex_by_rho = dp_mf_by_rho + dp_xc_by_rho
    p = ρ*T + p_ex
    
    results.append((T, ρ, p_ex, dp_ex_by_rho))

    if not args.output or args.verbose:
        print(f'{args.script}: model: {description}: T, T/Tc, ρ, ρ/ρc, p = {T:g}\t{T/Tc:g}\t{ρ:g}\t{ρ/ρc:g}\t{p:g}')

    if args.show:
        g = 1 + h # the pair function
        cut = r < args.rcut
        plt.plot(r[cut], g[cut])

if args.show:
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')
    plt.show()

if args.output:

    import pandas as pd

    df = pd.DataFrame(results, columns=['T', 'rho', 'p_ex', 'dp_ex_by_rho']).astype(float)
    
    df['T/Tc'] = df['T'] / Tc
    df['rho/rhoc'] = df['rho'] / ρc
    df['p'] = df['rho'] * df['T'] + df['p_ex']
    # df['dp'] = df['T'] + df['rho'] * df['dp_ex_by_rho']
    Δρ = pyHNC.grid_spacing(df['rho'].to_numpy())
    old_settings = np.seterr(divide='ignore')
    df['mu'] = log(df['rho']) + pyHNC.trapz_integrand(df['dp_ex_by_rho'].to_numpy(), dx=Δρ).cumsum()
    df = df[~np.isinf(df['mu'])] # remove floating point divergences

    with open(args.output, 'w') as f:
        f.write(f'# {description}\n')
        f.write(pyHNC.df_to_agr(df)) # use a utility here to convert to xmgrace format
        f.write('\n')

    print(f'{args.script}:', ', '.join(pyHNC.df_header(df)), f'> {args.output}')
