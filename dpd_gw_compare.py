#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The results of this calculation can be directly compared with Fig 4
# of the Groot and Warren [J. Chem. Phys. v107, 4423 (1997)].  The
# data from that figure is coded below.  This is taken from
# gw_p_compare.py from SunlightHNC.

# For the virial pressure here, see Eq. (2.5.22) in
# Hansen & McDonald, "Theory of Simple Liquids" (3rd edition):
# virial pressure, p = ρ - 2πρ²/3 ∫_0^∞ dr r³ dv/dr g(r) .

# The constant term here captures the mean field contribution, that
# is the integral evaluated with g(r) = 1.  Specifically:
# -∫_0^∞ dr r³ dv/dr g(r) = A ∫_0^1 dr r³(1-r) = A/20 .

# This program is part of pyHNC, copyright (c) 2023 Patrick B Warren
# Email: patrickbwarren@gmail.com

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

import numpy as np
import pandas as pd
from numpy import pi as π
from pyHNC import Grid, PicardHNC, truncate_to_zero
import matplotlib.pyplot as plt

gw_data = pd.DataFrame([[0.0, 0.0379935086163],
                        [1.5, 0.0751786298043],
                        [2.5, 0.0886823425022],
                        [3.0, 0.0924251622846],
                        [3.5, 0.0946639891655],
                        [4.0, 0.0965259421847],
                        [5.0, 0.0987451548125],
                        [6.0, 0.0998358473824],
                        [7.0, 0.1005510671090],
                        [8.0,  0.102017933031]],
                       columns=['rho', 'pexbyArho2'])

A = 25.0
ng, Δr = 4096, 0.01
grid = Grid(ng, Δr) # make the initial working grid
r = grid.r # extract the co-ordinate array for use below

vr = truncate_to_zero(A/2*(1-r)**2, r, 1.0) # DPD potential - the array here is size ng-1, same as r[:]
fr = truncate_to_zero((1-r), r, 1.0) # the derivate (negative), omitting the amplitude

solver = PicardHNC(grid)

results = []

for ρ in np.linspace(0.0, 10.0, 41)[1:]: # omit rho = 0.0
    soln = solver.solve(vr, ρ)
    hr = soln.hr
    pexbyArho2 = 2*π/3 * (1/20 + np.trapz(r**3*fr*hr, dx=Δr))
    results.append([ρ, pexbyArho2, solver.error])

hnc_data = pd.DataFrame(results, columns=['rho', 'pexbyArho2', 'error'])

# Make the data output suitable for plotting if captured by redirection
# stackoverflow.com/questions/30833409/python-deleting-the-first-2-lines-of-a-string

print('# ' + '\t'.join(hnc_data.columns))
hnc_data_s = '\n'.join(hnc_data.set_index('rho').to_string().split('\n')[2:])
print(hnc_data_s)

plt.plot(gw_data.rho, gw_data.pexbyArho2, 'ro', label='Groot & Warren (1997)')
plt.plot(hnc_data.rho, hnc_data.pexbyArho2, label='HNC')
plt.xlabel('$\\rho$')
plt.ylabel('$(p-\\rho)/A\\rho^2$')
plt.legend(loc='lower right')

plt.show()
