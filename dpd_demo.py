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

# Demonstrate the capabilities of the HNC package for solving DPD
# potentials, comparing with SunlightHNC if requested, and plotting
# the pair distribution function and the structure factor too.  For
# details here see also the SunlightHNC documentation.

# For standard DPD at A = 25 and ρ = 3, we have the following table

#           ∆t = 0.02   ∆t = 0.01   Monte-Carlo  HNC   deviation
# pressure  23.73±0.02  23.69±0.02  23.65±0.02   23.564  (0.4%)
# energy    13.66±0.02  13.64±0.02  13.63±0.02   13.762  (1.0%)
# mu^ex     12.14±0.02  12.16±0.02  12.25±0.10   12.171  (0.7%)

# The first two columns are from dynamic simulations.  The excess
# chemical potential (final row) is measured by Widom insertion in the
# simulations and calculated by SunlightHNC.  The pressure end energy
# density are from SunlightHNC and the present code and are in
# agreement to at least the indicated number of decimals.  The
# deviation is between HCNC and simulation results.

# Data is from a forthcoming publication on osmotic pressure in DPD.

import os
import pyHNC
import argparse
import numpy as np
from numpy import pi as π
from pyHNC import Grid, PicardHNC, truncate_to_zero

parser = argparse.ArgumentParser(description='DPD HNC calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser)
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('-A', '--A', default=25.0, type=float, help='repulsion amplitude, default 25.0')
parser.add_argument('-r', '--rho', default=3.0, type=float, help='density, default 3.0')
parser.add_argument('--dlambda', default=0.05, type=float, help='for coupling constant integration, default 0.05')
parser.add_argument('--rmax', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_argument('--qmax', default=25.0, type=float, help='maximum in q for plotting, default 25.0')
parser.add_argument('--sunlight', action='store_true', help='compare to SunlightHNC')
parser.add_argument('-s', '--show', action='store_true', help='show results')
args = parser.parse_args()

args.script = os.path.basename(__file__)

A, ρ = args.A, args.rho

grid = Grid(**pyHNC.grid_args(args)) # make the initial working grid

r, Δr, q = grid.r, grid.deltar, grid.q # extract the co-ordinate arrays for use below

if args.verbose:
    print(f'{args.script}: {grid.details}')

# Define the DPD potential, and its derivative, then solve the HNC
# problem.  The arrays here are all size ng-1, same as r[:]

φ = truncate_to_zero(A/2*(1-r)**2, r, 1) # the DPD potential
f = truncate_to_zero(A*(1-r), r, 1) # the force f = -dφ/dr

solver = PicardHNC(grid, **pyHNC.solver_args(args))

if args.verbose:
    print(f'{args.script}: {solver.details}')

soln = solver.solve(φ, ρ, monitor=args.verbose) # solve for the DPD potential
hr, hq = soln.hr, soln.hq # extract for use in a moment

# For the integrals here, see Eqs. (2.5.20) and (2.5.22) in Hansen &
# McDonald, "Theory of Simple Liquids" (3rd edition): for the (excess)
# energy density, e = 2πρ² ∫_0^∞ dr r² φ(r) g(r) and virial pressure,
# p = ρ + 2πρ²/3 ∫_0^∞ dr r³ f(r) g(r) where f(r) = −dφ/dr is the
# force.  An integration by parts shows that the mean-field
# contributions, being these with g(r) = 1, are the same.

# Here specifically the mean-field contributions are 
# 2πρ²/3 ∫_0^∞ dr r³ f(r) = A ∫_0^1 dr r³ (1−r) = πAρ²/30 .

e_mf = p_mf = π*A*ρ**2/30

e_xc = 2*π*ρ**2 * np.trapz(r**2*φ*hr, dx=Δr)
e_ex = e_mf + e_xc
e = 3*ρ/ + e_ex

p_xc = 2*π*ρ**2/3 * np.trapz(r**3*f*hr, dx=Δr)
p_ex = p_mf + p_xc
p = ρ + p_ex

# Coupling constant integration for the free energy

# Function to calculate the excess non-mean-field energy with coupling
# constant λ.  Uses a bunch of main script variables that are 'in
# scope' here.

def excess(λ):
    '''Return the excess correlation energy with coupling λ'''
    h = 0 if λ == 0 else solver.solve(λ*φ, ρ).hr # presumed will converge !
    e_xc = 2*π*ρ**2 * np.trapz(r**2*φ*h, dx=Δr) # the integral above
    if args.verbose and args.verbose > 1:
        print(f'{args.script}: excess: λ = {λ:0.3f}, e_xc = {e_xc:f}')
    return e_xc

λ_arr = np.linspace(0, 1, 1+round(1/args.dlambda))
dλ = pyHNC.grid_spacing(λ_arr)
e_xc_arr = np.array([excess(λ) for λ in np.flip(λ_arr)]) # descend, to assure convergence
f_xc = np.trapz(e_xc_arr, dx=dλ) # the coupling constant integral
f_ex = e_mf + f_xc # f_mf = e_mf in this case

print(f'{args.script}: model: standard DPD with A = {A:g}, ρ = {ρ:g}')

print(f'{args.script}: Monte-Carlo (A,ρ = 25,3):      energy, virial pressure =\t13.63±0.02\t\t\t23.65±0.02')
print(f'{args.script}: pyHNC v{pyHNC.version}:        energy, free energy, virial pressure =',
      '\t%0.5f\t%0.5f\t%0.5f' % (e_ex, f_ex, p))

if args.sunlight:
    
    from oz import wizard as w

    w.ng = grid.ng
    w.deltar = grid.deltar

    w.initialise()
    w.arep[0,0] = A
    w.dpd_potential()
    w.rho[0] = ρ
    w.hnc_solve()
    
    sunlight_version = str(w.version, 'utf-8').strip()
    print(f'{args.script}: SunlightHNC v{sunlight_version}: energy, free energy, virial pressure =',
          '\t%0.5f\t%0.5f\t%0.5f' % (w.uex, w.aex, w.press))

if args.show:

    import matplotlib.pyplot as plt

    gr = 1.0 + hr # the pair function
    sq = 1.0 + ρ*hq # the structure factor

    plt.figure(1)
    cut = r < args.rmax
    if args.sunlight:
        imax = int(args.rmax / w.deltar)
        plt.plot(w.r[0:imax], 1.0+w.hr[0:imax,0,0], '.')
        plt.plot(r[cut], gr[cut], '--')
    else:
        plt.plot(r[cut], gr[cut])        
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')

    plt.figure(2)
    cut = q < args.qmax
    if args.sunlight:
        jmax = int(args.qmax / w.deltak)
        plt.plot(w.k[0:jmax], w.sk[0:jmax,0,0]/ρ, '.')
        plt.plot(q[cut], sq[cut], '--')
    else:
        plt.plot(q[cut], sq[cut])
    plt.xlabel('$k$')
    plt.ylabel('$S(k)$')

    plt.show()
