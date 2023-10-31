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

# Generate EoS data for standard DPD

import pyHNC
import argparse
import numpy as np
import pandas as pd
from numpy import pi as π
from pyHNC import truncate_to_zero

parser = argparse.ArgumentParser(description='DPD EoS calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser)
parser.add_argument('--Arange', action='store', default='10(10)50', help='repulsion amplitude range')
parser.add_argument('--rhorange', action='store', default='1(1)10', help='density range')
args = parser.parse_args()

grid = pyHNC.Grid(**pyHNC.grid_args(args)) # make the initial working grid
r, Δr = grid.r, grid.deltar # extract the co-ordinate array for use below

solver = pyHNC.PicardHNC(grid, **pyHNC.solver_args(args))

# DPD potential and force law omitting amplitude;
# the array sizes here are ng-1, same as r[:].

vr = truncate_to_zero(1/2*(1-r)**2, r, 1.0)
fr = truncate_to_zero((1-r), r, 1.0)

# The virial pressure, p = ρ + 2πρ²/3 ∫_0^∞ dr r³ f(r) g(r) where
# f(r) = −dv/dr is the force.  See Eq. (2.5.22) in Hansen & McDonald,
# "Theory of Simple Liquids" (3rd edition).  The constant term is the
# mean field contribution, that is the integral evaluated with g(r) = 1,
# namely ∫_0^∞ dr r³ f(r) = A ∫_0^1 dr r³ (1−r) = A/20.

data = []
for A in pyHNC.as_linspace(args.Arange):
    solver.warmed_up = False # fresh start with lowest density
    for ρ in pyHNC.as_linspace(args.rhorange):
        hr = solver.solve(A*vr, ρ).hr
        pexbyA = 2*π*ρ**2/3 * (1/20 + np.trapz(r**3*fr*hr, dx=Δr))
        p = ρ + A*pexbyA
        data.append((A, ρ, ρ**2, p, pexbyA, solver.error))

df = pd.DataFrame(data, columns=['A', 'rho', 'rhosq', 'p', 'pexbyA', 'error'])
print(pyHNC.df_to_agr(df))
