#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Demonstrate the capabilities of the HNC package for solving DPD
# potentials, comparing with SunlightHNC if requested, and plotting
# the pair distribution function and the structure factor too.  For
# details here see also the SunlightHNC documentation.

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

import argparse
import numpy as np
from pyHNC import Grid, PicardHNC, truncate_to_zero
from numpy import pi as π

parser = argparse.ArgumentParser(description='DPD HNC calculator')
parser.add_argument('-v', '--verbose', action='store_true', help='report convergence')
parser.add_argument('--ngrid', action='store', default='2^12', help='number of grid points, default 2^12 = 4096')
parser.add_argument('--deltar', action='store', default=1e-2, type=float, help='grid spacing, default 1e-2')
parser.add_argument('--A', action='store', default=25.0, type=float, help='repulsion amplitude, default 25.0')
parser.add_argument('--rho', action='store', default=3.0, type=float, help='density, default 3.0')
parser.add_argument('--alpha', action='store', default=0.2, type=float, help='Picard mixing fraction, default 0.2')
parser.add_argument('--picard', action='store', default=500, type=int, help='max number of Picard steps, default 500')
parser.add_argument('--tol', action='store', default=1e-12, type=float, help='tolerance for convergence, default 1e-12')
parser.add_argument('--rmax', action='store', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_argument('--qmax', action='store', default=25.0, type=float, help='maximum in q for plotting, default 25.0')
parser.add_argument('--sunlight', action='store_true', help='compare to SunlightHNC')
parser.add_argument('--show', action='store_true', help='show results')
args = parser.parse_args()

A, ρ = args.A, args.rho

print('Model: standard DPD with A = %f, ρ = %f' % (A, ρ))

ng = eval(args.ngrid.replace('^', '**')) # catch 2^10 etc
Δr = args.deltar
grid = Grid(ng, Δr) # make the initial working grid
r, q = grid.r, grid.q # extract the co-ordinate arrays for use below

# Define the canonical unnormalised weight function used in the DPD
# potential, and its derivative, then solve the HNC problem.

wr = truncate_to_zero(1/4*(1-r)**2, r, 1.0) # the array here is size ng-1, same as r[:]
minusdwdr = truncate_to_zero(1/2*(1-r), r, 1.0) # the derivate (negative)

solver = PicardHNC(grid, alpha=args.alpha, tol=args.tol, max_iter=args.picard, monitor=args.verbose)
soln = solver.solve(2*A*wr, ρ) # solve for the DPD potential, being 2A × the weight function
hr, hq = soln.hr, soln.hq # extract for use in a moment

# For the integrals here, see Eqs. (2.5.20) and (2.5.22) in
# Hansen & McDonald, "Theory of Simple Liquids" (3rd edition):
# energy density, e = 2πρ² ∫_0^∞ dr r² v(r) g(r) ;
# virial pressure, p = ρ - 2πρ²/3 ∫_0^∞ dr r³ dv/dr g(r) .

# The constant terms here capture the mean field contributions, that
# is the integrals evaluated with g(r) = 1.  Specifically:
# ∫_0^∞ dr r² v(r) g(r) = A ∫_0^1 dr r²(1-r)²/2 = A/60 ;
# -∫_0^∞ dr r³ dv/dr g(r) = A ∫_0^1 dr r³(1-r) = A/20 .

energy = 2*π*ρ**2 * (A/60 + 2*A*np.trapz(r**2*wr*hr, dx=Δr))
pressure = ρ + 2*π*ρ**2/3 * (A/20 + 2*A*np.trapz(r**3*minusdwdr*hr, dx=Δr))

print('pyHNC v%s:    virial pressure, energy density =\t\t%0.5f\t%0.5f' % (grid.version, pressure, energy))

if args.sunlight:
    
    from oz import wizard as w

    w.initialise()
    w.arep[0,0] = A
    w.dpd_potential()
    w.rho[0] = ρ
    w.hnc_solve()
    
    version = str(w.version, 'utf-8').strip()
    print('SunlightHNC v%s: virial pressure, energy density =\t\t%0.5f\t%0.5f' % (version, w.press, w.uex))

if args.show:

    import matplotlib.pyplot as plt

    gr = 1.0 + hr # the pair function
    sq = 1.0 + ρ*hq # the structure factor

    plt.figure(1)
    cut = r < args.rmax
    if args.sunlight:
        imax = int(3.0 / w.deltar)
        plt.plot(w.r[0:imax], 1.0+w.hr[0:imax,0,0], '.')
        plt.plot(r[cut], gr[cut], '--')
    else:
        plt.plot(r[cut], gr[cut])        
    plt.xlabel('$r$')
    plt.ylabel('$g(r)$')

    plt.figure(2)
    cut = q < args.qmax
    if args.sunlight:
        jmax = int(25.0 / w.deltak)
        plt.plot(w.k[0:jmax], w.sk[0:jmax,0,0]/ρ, '.')
        plt.plot(q[cut], sq[cut], '--')
    else:
        plt.plot(q[cut], sq[cut])
    plt.xlabel('$k$')
    plt.ylabel('$S(k)$')

    plt.show()
