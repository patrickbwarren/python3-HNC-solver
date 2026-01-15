#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This program is part of pyHNC, copyright (c) 2023-2025
# Patrick B Warren (STFC).
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

# Use the HNC package to solve the nDPD equation of state in the high
# temperature approximation to the Weeks-Chandler-Andersen (WCA)
# theory.  The nDPD model is described in Sokhan et al., Soft Matter
# 19, 5824 (2023).

# To lowest order the system has a van der Waals loop for at least
## ./ndpd_wca.py -n 4 -T 1.08
## ./ndpd_wca.py -n 2 -T 1.0

import os
import pyHNC
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import pi as π
from pyHNC import Grid, PicardHNC, truncate_to_zero, ExtendedArgumentParser
from scipy.integrate import simpson, cumulative_simpson
from numpy import log as ln

parser = ExtendedArgumentParser(description='nDPD HNC WCA calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser, alpha=0.2, npicard=1000) # increase number of Picard steps
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('-n', '--n', default='2', help='governing exponent, default 2')
parser.add_argument('-A', '--A', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-B', '--B', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-T', '--T', default='1.08,1.11,4', help='temperature range, default 1.08,1.11,4')
parser.add_argument('-r', '--rho', default='1,8,71', help='density range, default 1,8,71')
parser.add_argument('-m', '--mu', action='store_true', help='plot p(mu) rather than p(V)')
parser.add_argument('-s', '--show', action='store_true', help='show results')
parser.add_argument('-o', '--output', help='write pair function to a file')
args = parser.parse_args()

args.script = os.path.basename(__file__)

# The following are Tables from Sokhan et al., Soft Matter 19, 5824 (2023).

table1 = {'2': (2, 25.0,  3.02), # n, A, B, values
          '3': (3, 15.0,  7.2),
          '4': (4, 10.0, 15.0)}

table2 = {'2': (1.025, 0.2951, 0.519), # T_c, p_c, ρ_c values
          '3': (1.283, 0.3979, 0.504),
          '4': (1.290, 0.4095, 0.484)}

if args.n in table1:
    n, A, B = table1[args.n] # unpack the default values
    Tc, pc, ρcσ3 = table2[args.n] # where available
else:
    raise ValueError(f'{args.script}: currently n is restricted to ' + ', '.join(table1.keys()))

A = args.A if args.A is not None else A # overwrite if necessary
B = args.B if args.B is not None else B # overwrite if necessary

σ = 1 - ((n+1)/(2*B))**(1/(n-1)) # the size is where the potential vanishes

R = 1 - B**(-1/(n-1)) # the minimum in the potential
ε = - A*B/(n+1)*(1-R)**(n+1) + A/2*(1-R)**2 # (minus) the potential at the minimum

ρc = ρcσ3 / σ**3 # back out the critical value

grid = Grid(**pyHNC.grid_args(args)) # make the initial working grid

r, q = grid.r, grid.q # extract the co-ordinate arrays for use below

if args.verbose:
    print(f'{args.script}: {grid.details}')

# Define the nDPD potential as in Eq. (5) in Sokhan et al., assuming
# r_c = 1, and the derivative, then solve the HNC problem.  The arrays
# here are all size ng-1, same as r[:]

u = truncate_to_zero(A*B/(n+1)*(1-r)**(n+1) - A/2*(1-r)**2, r, 1) # the nDPD potential
f = truncate_to_zero(A*B*(1-r)**n - A*(1-r), r, 1) # the corresponding force f = -dφ/dr

# Implement the WCA split 

urep = truncate_to_zero(u+ε, r, R)
frep = truncate_to_zero(f, r, R)
uatt = u - urep

solver = PicardHNC(grid, nmonitor=500, **pyHNC.solver_args(args))

if args.verbose: 
    print(f'{args.script}: {solver.details}')

ρ_arr = pyHNC.as_linspace(args.rho)
T_arr = pyHNC.as_linspace(args.T)

data = []

for T in T_arr:
    β = 1 / T
    for ρ in ρ_arr:
        soln = solver.solve(β*urep, ρ, monitor=args.verbose)
        if not solver.converged:
            continue
        gref = 1 + soln.hr
        p_ref = 2/3*π*ρ**2 * simpson(r**3*frep*gref, r) # excess pressure in reference
        φ = -β*uatt # to match Andrew's note
        a_hta = -2*π*ρ*simpson(r**2*gref*φ, r) # high temperature approximation
        datum = (T, ρ, p_ref, a_hta)
        data.append(datum)

df = pd.DataFrame(data, columns=['T', 'ρ', 'p_ref', 'a_hta'])

if args.show:

    ρ = ρ_arr

    for T in T_arr:
        df2 = df[df['T']==T]
        p_ref = df2.p_ref
        μ_ref = cumulative_simpson(1/ρ, x=p_ref, initial=0)
        a_hta = df2.a_hta
        p_hta = ρ**2 * np.gradient(a_hta, ρ)
        μ_hta = a_hta + ρ*np.gradient(a_hta, ρ)
        μ = T*ln(ρ) + μ_ref + μ_hta
        p = ρ*T + p_ref + p_hta
        x = μ if args.mu else 1/(ρ*σ**3)
        plt.plot(x, p, label=f'$T = {T:0.2f}$')

    xlabel = r'$\mu$' if args.mu else r'$V/N\sigma^3$'
    plt.xlabel(xlabel)
    plt.ylabel(r'$P$')
    plt.legend()
    plt.show()

if args.output:

    df_agr = pyHNC.df_to_agr(df)
    with open(args.output, 'w') as f:
        f.write(df_agr)
        f.write('\n')
    print(f'Written data to {args.output}')
