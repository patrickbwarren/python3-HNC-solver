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

# Generate tables of EoS data for standard DPD.

import pyHNC
import argparse
import numpy as np
import pandas as pd
from numpy import pi as π
from pyHNC import truncate_to_zero

parser = argparse.ArgumentParser(description='DPD EoS calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser)
parser.add_argument('-A', '--A', default='10(10)50', help='repulsion amplitude range')
parser.add_argument('-r', '--rho', default='1(1)10', help='density range')
args = parser.parse_args()

grid = pyHNC.Grid(**pyHNC.grid_args(args)) # make the initial working grid
r, Δr = grid.r, grid.deltar # extract the co-ordinate array for use below

solver = pyHNC.PicardHNC(grid, **pyHNC.solver_args(args))

# DPD potential and force law omitting amplitude;
# the array sizes here are ng-1, same as r[:].

φbyA = truncate_to_zero(1/2*(1-r)**2, r, 1)
fbyA = truncate_to_zero((1-r), r, 1)
w = 15/π * φbyA # normalised weight function

# The virial pressure, p = ρ + 2πρ²/3 ∫_0^∞ dr r³ f(r) g(r) where
# f(r) = −d φ/dr is the force.  See Eq. (2.5.22) in Hansen & McDonald,
# "Theory of Simple Liquids" (3rd edition).

# The constant term is the mean field contribution, namely
# 2πρ²/3 ∫_0^∞ dr r³ f(r) = A ∫_0^1 dr r³ (1−r) = πAρ²/30.

# We calculate also <n> = 4π ∫_0^∞ dr r² w(r) g(r).

data = [] # this will grow as computations proceed

for A in pyHNC.as_linspace(args.A):
    solver.warmed_up = False # fresh start with lowest density
    for ρ in pyHNC.as_linspace(args.rho):
        h = solver.solve(A*φbyA, ρ).hr # just keep h(r)
        if solver.converged: # but do test if converged !
            pexbyA = π*ρ**2/30 + 2*π*ρ**2/3 * np.trapz(r**3*fbyA*h, dx=Δr)
            ζav = 4*π * np.trapz(r**2*w*h, dx=Δr) # may notation-clash with mbdpd codes
            nav = ρ*(1 + ζav) # the mean local density
            p = ρ + A*pexbyA
            data.append((A, ρ, ρ**2, p, pexbyA, ζav, nav, solver.error))

df = pd.DataFrame(data, columns=['A', 'rho', 'rhosq', 'p', 'pexbyA', '<xi>', '<n>', 'error'])

print(pyHNC.df_to_agr(df)) # use a utility here to convert to xmgrace format
