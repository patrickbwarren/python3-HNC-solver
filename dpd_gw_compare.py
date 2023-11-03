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

# The results of this calculation can be directly compared with Fig 4
# of the Groot and Warren [J. Chem. Phys. v107, 4423 (1997)].  The
# data from that figure is coded below.  This is taken from
# gw_p_compare.py from SunlightHNC.

# For the virial pressure here, see Eq. (2.5.22) in Hansen & McDonald,
# "Theory of Simple Liquids" (3rd edition): virial pressure, p = ρ +
# 2πρ²/3 ∫_0^∞ dr r³ f(r) g(r) where f(r) = −dv/dr is the force.

# The constant term here captures the mean field contribution, that
# is the integral evaluated with g(r) = 1.  Specifically:
# ∫_0^∞ dr r³ f(r) = A ∫_0^1 dr r³ (1−r) = A/20 .

import numpy as np
import pandas as pd
from numpy import pi as π
from pyHNC import Grid, PicardHNC, truncate_to_zero, df_to_agr
import matplotlib.pyplot as plt

gw_data_df = pd.DataFrame([[0.0, 0.038], [1.5, 0.075], [2.5, 0.089], [3.0, 0.092], [3.5, 0.095],
                           [4.0, 0.097], [5.0, 0.099], [6.0, 0.100], [7.0, 0.101], [8.0, 0.102]],
                          columns=['rho', 'pexbyArho2'])

A = 25.0
Δr, ng = 0.02, 8192
grid = Grid(ng, Δr) # make the initial working grid
r = grid.r # extract the co-ordinate array for use below

φ = truncate_to_zero(A/2*(1-r)**2, r, 1) # DPD potential
fbyA = truncate_to_zero((1-r), r, 1) # the force f(r) = −dφ/dr, omitting the amplitude

solver = PicardHNC(grid)

hnc_data = []
for ρ in np.linspace(0.0, 10.0, 41)[1:]: # omit rho = 0.0
    hr = solver.solve(φ, ρ).hr
    pexbyArho2 = 2*π/3 * (1/20 + np.trapz(r**3*fbyA*hr, dx=Δr))
    p = ρ+ A*ρ**2*pexbyArho2
    hnc_data.append([ρ, p, pexbyArho2, solver.error])

hnc_data_df = pd.DataFrame(hnc_data, columns=['rho', 'p', 'pexbyArho2', 'error'])
print(df_to_agr(hnc_data_df))

plt.plot(gw_data_df.rho, gw_data_df.pexbyArho2, 'ro', label='Groot & Warren (1997)')
plt.plot(hnc_data_df.rho, hnc_data_df.pexbyArho2, label='HNC')
plt.xlabel('$\\rho$')
plt.ylabel('$(p-\\rho)/A\\rho^2$')
plt.legend(loc='lower right')

plt.show()
