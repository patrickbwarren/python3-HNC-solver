#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This program is part of pyHNC, copyright (c) 2023 Patrick B Warren (STFC).
# Additional modifications copyright (c) 2025 Joshua F Robinson (STFC).  
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

# Hyper-netted chain (HNC) solver for Ornstein-Zernike (OZ) equation.

import pyfftw
import argparse
import numpy as np
from scipy.integrate import simpson
from collections import deque
from .utilities import *


# Provide a grid as a working platform.  This is the pair of arrays
# r(:) and q(:) initialised to match the desired ng (# grid points)
# and Δr.  Note that the array lengths are actually ng-1.  A real odd
# discrete Fourier transform (RODFT00) is also initialised, with
# functions to do forward and backward Fourier-Bessel transforms
# (radial 3d). For implementation details of FFTW see fftw_test.py.
# In particular note that Δr×Δk = π / (n+1) where n is the length of
# the FFT arrays.  Testing indicate the best efficiency is obtained
# when ng = 2^r with r being an integer.

class RadialGrid:

    def __init__(self, ng=8192, deltar=0.02):
        '''Initialise grids with the desired size and spacing'''
        self.ng = ng
        self.deltar = deltar
        self.deltaq = np.pi / (self.deltar*self.ng) # as above
        self.r = self.deltar * np.arange(1, self.ng) # start from 1, and of length ng-1
        self.q = self.deltaq * np.arange(1, self.ng) # ditto
        self.fftwx = pyfftw.empty_aligned(self.ng-1)
        self.fftwy = pyfftw.empty_aligned(self.ng-1)
        self.fftw = pyfftw.FFTW(self.fftwx, self.fftwy,
                                direction='FFTW_RODFT00',
                                flags=('FFTW_ESTIMATE',))
        self.parstrings = [f'ng = {self.ng} = 2^{round(np.log2(ng))}',
                           f'Δr = {self.deltar}', f'Δq = {self.deltaq:0.3g}',
                           f'|FFTW arrays| = {self.ng-1}']
        self.details = 'Grid: ' + ', '.join(self.parstrings)

    # These functions assume the FFTW has been initialised as above, the
    # arrays r and q exist, as do the parameters Δr and Δq.

    def fourier_bessel_forward(self, fr):
        '''Forward transform f(r) to reciprocal space'''
        self.fftwx[:] = self.r * fr
        self.fftw.execute()
        return 2*np.pi*self.deltar/self.q * self.fftwy

    def fourier_bessel_backward(self, fq):
        '''Back transform f(q) to real space'''
        self.fftwx[:] = self.q * fq
        self.fftw.execute()
        return self.deltaq/(4*np.pi**2*self.r) * self.fftwy

# Assume radial by default as normally spherical polars will be used for
# spherical pair potentials
Grid = RadialGrid

# What's being solved here is the Ornstein-Zernike (OZ) equation in
# the form h(q) = c(q) + ρ h(q) c(q) in combination with the HNC
# closure g(r) = exp[ - v(r) + h(r) - c(r)] iteratively.
# Here c(r) is the direct correlation function, h(r) = g(r) - 1 is the
# total correlation function, and v(r) is the potential.  In practice
# the OZ equation and the HNC closure are written in terms of the
# indirect correlation function e(r) = h(r) - c(r).  An initial guess
# if the solver is not warmed up is c(r) = - v(r) (ie, the RPA soln).

class Solver:

    def __init__(self, grid,
                 alpha=0.5, tol=1e-12, niters=500, npicard=None,
                 history_size=4, line_search_decay=0.8, nline_searches=25,
                 nmonitor=50):
        '''Initialise basic data structure'''
        self.grid = grid

        self.alpha = alpha
        self.tol = tol
        self.niters = niters
        self.history_size = 4
        if npicard is None: npicard = history_size + 1
        self.npicard = npicard
        self.line_search_decay = line_search_decay
        self.nline_searches = nline_searches

        self.nmonitor = nmonitor

        self.converged = False
        self.warmed_up = False
        self.parstrings = [f'α = {self.alpha}', f'tol = {self.tol:0.1e}',
                           f'niters = {self.niters}']
        self.name = 'Solver'
        self.details = f'{self.name}: ' + ', '.join(self.parstrings)

    def oz_solution(self, rho, cq):
        '''Solve the OZ equation for h in terms of c, in reciprocal space.'''
        return cq / (1 - rho*cq)

    def solve(self, vr, rho, cr_init=None, monitor=False):
        '''Solve HNC for a given potential, with an optional initial guess at cr'''
        self_name = self.name + '.solve' # used in reporting below
        cr = np.copy(cr_init) if cr_init is not None else np.copy(self.cr) if self.warmed_up else -np.copy(vr)
        expnegvr = np.exp(-vr) # for convenience, also works with np.inf

        inner = lambda u, v: simpson(u*v, dx=self.grid.deltar)
        magnitude = lambda u: inner(u, u)**0.5

        # Memory of iterations for inferring hessian
        cr_in = deque(maxlen=self.history_size) # input value of c
        cr_out = deque(maxlen=self.history_size) # output value of c
        cr_delta = deque(maxlen=self.history_size) # change out[i] - in[i]

        # Solve OZ equation during each iteration.
        def iteration(cr):
            if np.any(np.isnan(cr)): raise ValueError
            cq = self.grid.fourier_bessel_forward(cr) # forward transform c(r) to c(q)
            eq = self.oz_solution(rho, cq) - cq # solve the OZ equation for e(q) = h(q) - c(q)
            er = self.grid.fourier_bessel_backward(eq) # back transform e(q) to e(r)
            cr_new = expnegvr*np.exp(er) - er - 1 # iterate with the HNC closure
            if np.any(np.isnan(cr_new)): raise ValueError
            return cr_new, er

        cr_in += [cr.copy()]
        cr_new, er = iteration(cr)
        cr_out += [cr_new.copy()]
        cr_delta += [cr_out[-1] - cr_in[-1]]

        for iter in range(self.niters):

            # Determine optimal direction for line search.
            if iter < self.npicard:
                # First few iterations we do not have curvature information
                # so we must resort to normal downhill direction (Picard). 
                step = cr_delta[-1]
                step_size = self.alpha

            else:
                # Switch to limited-memory quasi-Newton method of
                # K.-C. Ng, J. Chem. Phys. 61, 2680 (1974).
  
                n = len(cr_in)
                delta = [cr_delta[-1] - cr_delta[-i-1] for i in range(1, n)]
                residual = [inner(cr_delta[-1], delta[i]) for i in range(n-1)]

                A = np.empty((n-1, n-1))
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        A[i,j] = inner(delta[i], delta[j])
                        A[j,i] = A[i,j]

                α = np.zeros(n)
                α[1:] = np.linalg.solve(A, residual)
                α[0] = 1 - np.sum(α[1:])
                α = np.flipud(α)
                step = α.dot(cr_out) - cr_in[-1]
                step_size = 1.

            if np.any(np.isnan(step)): raise ValueError

            # Backtracking line search
            for line_iter in range(self.nline_searches):
                cr = cr_in[-1] + step_size * step

                self.error = np.inf
                try:
                    cr_new, er = iteration(cr)
                    if np.any(np.isnan(cr_new)): raise ValueError
                    delta = cr_new - cr
                    self.error = magnitude(delta)

                except: pass

                if np.isfinite(self.error): break
                step_size *= self.line_search_decay

            else:
                raise RuntimeError('line search stalled!')

            cr_in += [cr.copy()]
            cr_out += [cr_new.copy()]
            cr_delta += [delta.copy()]

            # Convergence test
            self.converged = self.error < self.tol
            if monitor and (iter % self.nmonitor == 0 or self.converged):
                iter_s = f'{self_name}: iteration %{len(str(self.niters))}d,' % iter
                print(f'{iter_s} error = {self.error:0.3e}')
            if self.converged:
                break

        if self.converged:
            self.cr = cr_new # use the most recent calculation
            self.cq = self.grid.fourier_bessel_forward(cr)
            self.hr = self.cr + er # total correlation function
            eq = self.grid.fourier_bessel_forward(er)
            self.hq = self.cq + eq
            self.warmed_up = True
        else: # we leave it to the user to check if self.converged is False :-)
            pass
        if monitor:
            if self.converged:
                print(f'{self_name}: converged')
            else:
                print(f'{self_name}: iteration {iter:3d}, error = {self.error:0.3e}')
                print(f'{self_name}: failed to converge')
        return self # the user can name this 'soln' or something

# Below, the above is sub-classed to redefine the OZ equation in terms
# of the product of the solvent structure factor S(q).  This enables
# the above machinery to be re-used for solving the problem of an
# infinitely dilute solute inside the solvent.

# The math here is as follows.  In a two-component system the OZ equations are
#   h00q = c00q + rho0 c00q h00q + rho1 c01q h01q,
#   h01q = c01q + rho0 c01q h00q + rho1 c11q h01q,
#   h01q = c01q + rho0 c00q h01q + rho1 c01q h11q,
#   h11q = c11q + rho0 c01q h01q + rho1 c11q h11q.
# (the equivalence of the two off-diagonal expressions can be verified).
# In the limit rho1 --> 0 the second component becomes an infinitely
# dilute solute in the first component (solvent).
# The OZ relations in this limit are
#   h00q = c00q + rho0 c00q h00q,
#   h01q = c01q + rho0 c01q h00q
#   h01q = c01q + rho0 c00q h01q,
#   h11q = c11q + rho0 c01q h01q.
# The first of these is simply the one-component OZ relation for the solvent.
# The second of these can be written as
#   h01q = c01q S00q
# where S00q = 1 + rho0 h00q is the solvent structure factor This
# should be supplemented by the HNC closure in the off-diagonal
# component.  To solve this case therefore we ask the user to provide
# S00q, and change the OZ relation that the solver uses.  This is what
# is implemented below.

# This solver class can also be repurposed to solve the mean-field DFT
# problem in Archer and Evans, J. Chem. Phys. 118(21), 9726-46 (2003).

# The governing equation corresponds to eq (15) in the above paper and
# describes the density of solvent particles around a test solute
# particle.  This density can be written as rho0 g01 and eq (15) can
# be cast into the form ln g01 = - v01 - rho0 h01 * v00 where '*'
# denotes a convolution and h01 = g01 - 1.  Given a solution to this,
# the solvent-mediated potential between the test particle and a
# second particle is expressed in eq (10) in the above paper which can
# be written W12 = rho0 h01 * v02.  Given the privileged role of the
# test particle it is apparent that this approach doesn't necessarily
# satisfy reciprocity W12 = W21 (discussed in Archer + Evans), but one
# might hope the deviations are small.

# If we define e01 = - rho0 h01 * v00 and c01 = h01 - e01, the
# governing equation can be written as the pair
#  h01q = c01q / (1 + rho0 v00q),
#  ln g01 = - v01 + e01.
# In this form they strongly resemble the problem of the infinitely
# dilute solute in HNC solved above, with the only change being to the
# replace the OZ relation.  Note that c01 and e01 as defined are NOT
# the direct and indirect correlation functions since this mean-field
# DFT approach is an RPA-HNC hybrid in some sense.

# To utilise the code for this problem instantiate SoluteSolver
# using 1 / (1 + rho0 v00q) instead of S00q, or replace this in an
# existing instantiation.

# Finally the solver class can also be repurposed to solve the vanilla
# RISM equations for homodimers.  The RISM eqs H = Ω.C.Ω + Ω.C.R.H
# closed by HNC reduce in the case of infinitely dilute heterodimers
# to the standard HNC problem for the solvent, plus the following for
# the site-solvent functions
#  h01q = (c01q + omega12q c02q) (1 + rho0 h00q),
#  h02q = (c02q + omega12q c01q) (1 + rho0 h00q),
# where omega12q = sin(ql) / (ql) for a rigid bond.  From these it is
# clear that in the homodimer case, utilising S00q = 1 + rho0 h00q,
#  h01q = h02q = c01q S00q (1 + omega12q).
# This is in the form required to repurpose the solute OZ relation.

# To utilise the code for this problem instantiate SoluteSolver
# using S00q (1 + omega12q) instead of S00q, or replace this in an
# existing instantiation.

class SoluteSolver(Solver):
    '''Subclass for infinitely dilute solute inside solvent.'''

    def __init__(self, S00q, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S00q = S00q
        self.name = 'SoluteSolver'
        self.details = f'{self.name}: ' + ', '.join(self.parstrings)

    def oz_solution(self, rho, cq): # rho is not used here
        '''Solve the modified OZ equation for h, in reciprocal space.'''
        return self.S00q * cq

    def solve(self, vr, cr_init=None, monitor=False):
        return super().solve(vr, 0.0, cr_init, monitor) # rho = 0.0 is not needed

# Below, cases added by Josh

class TestParticleRPA(Solver):
    '''Subclass for mean-field DFT approach.'''
    
    def oz_solution(self, rho, cq):
        '''Solution to the OZ equation in reciprocal space.'''
        return cq / (1 + rho*self.vq) # force RPA closure in reciprocal term

    def solve(self, vr, *args, **kwargs):
        self.vq = self.grid.fourier_bessel_forward(vr) # forward transform v(r) to v(q)
        return super().solve(vr, *args, **kwargs)

class SoluteTestParticleRPA(SoluteSolver):
    
    def oz_solution(self, rho, cq):
        '''Solution to the OZ equation in reciprocal space.'''
        return cq - (self.S00q - 1) * self.vq01 # RPA closure

    def solve(self, vr01, *args, **kwargs):
        self.vq01 = self.grid.fourier_bessel_forward(vr01) # forward transform v(r) to v(q)
        return super().solve(vr01, *args, **kwargs)
