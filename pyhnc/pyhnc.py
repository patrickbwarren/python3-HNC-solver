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
from . import potentials

from abc import ABC, abstractmethod
from typing import Type
from numpy.typing import NDArray

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
        """Initialise grids with the desired size and spacing"""
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

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
        return f'{self.name}: ' + ', '.join(self.parstrings)

    # These functions assume the FFTW has been initialised as above, the
    # arrays r and q exist, as do the parameters Δr and Δq.

    def fourier_bessel_forward(self, fr):
        """Forward transform f(r) to reciprocal space"""
        self.fftwx[:] = self.r * fr
        self.fftw.execute()
        return 2*np.pi*self.deltar/self.q * self.fftwy

    def fourier_bessel_backward(self, fq):
        """Back transform f(q) to real space"""
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

class OrnsteinZernikeSolver(ABC):

    @classmethod
    def from_instance(cls, old):
        new = cls(**old.init_kwargs())

        new.converged = old.converged
        new.warmed_up = old.warmed_up

        if old.warmed_up:
            new.e = old.e.copy()
            new.b = old.b.copy()
            new.c = old.c.copy()
            new.potential = old.potential
            new.rho = old.rho
            new.T = old.T
            new.error = old.error

        return new

    def copy(self):
        return self.from_instance(self)

    def init_kwargs(self):
        return {'grid': self.grid,
                'alpha': self.alpha,
                'tol': self.tol,
                'niters': self.niters,
                'npicard': self.npicard,
                'history_size': self.history_size,
                'line_search_decay': self.line_search_decay,
                'nline_searches': self.nline_searches,
                'nmonitor': self.nmonitor}

    def __init__(self, grid,
                 alpha=0.5, tol=1e-12, niters=500, npicard=None,
                 history_size=4, line_search_decay=0.8, nline_searches=25,
                 nmonitor=50):
        """Initialise basic data structure"""
        self.grid = grid

        self.alpha = alpha
        self.tol = tol
        self.niters = niters
        self.history_size = history_size
        if npicard is None: npicard = history_size + 1
        self.npicard = npicard
        self.line_search_decay = line_search_decay
        self.nline_searches = nline_searches

        self.nmonitor = nmonitor

        self.converged = False
        self.warmed_up = False
        self.parstrings = [f'α = {self.alpha}', f'tol = {self.tol:0.1e}',
                           f'niters = {self.niters}']

    @property
    def name(self):
        return type(self).__name__

    def __repr__(self):
        return f'{self.name}: ' + ', '.join(self.parstrings)

    @property
    def r(self):
        return self.grid.r

    @property
    def density(self):
        return self.rho

    @property
    def h(self):
        return self.e + self.c

    @property
    def g(self):
        return 1 + self.h

    @property
    def eq(self):
        return self.grid.fourier_bessel_forward(self.e)

    @property
    def bq(self):
        return self.grid.fourier_bessel_forward(self.b)

    @property
    def cq(self):
        return self.grid.fourier_bessel_forward(self.c)

    @property
    def hq(self):
        return self.grid.fourier_bessel_forward(self.h)

    @property
    def gq(self):
        return self.grid.fourier_bessel_forward(self.g)

    @property
    def Sq(self):
        return 1 + self.rho * self.hq

    @abstractmethod
    def bridge_closure(self, e: NDArray, *args, **kwargs):
        """Closure of the OZ equation for $b(r)$."""
        raise NotImplementedError

    def oz_solution_hq_from_cq(self, cq: NDArray, rho: float, *args, **kwargs):
        """Solve the reciprocal OZ equation for $h(q)$ in terms of $c(q)$."""
        return cq / (1 - rho*cq)

    def oz_solution_cq_from_hq(self, hq: NDArray, rho: float, *args, **kwargs):
        """Solve the reciprocal OZ equation for $c(q)$ in terms of $h(q)$."""
        return hq / (1 + rho*hq)

    def oz_solution_eq_from_cq(self, cq: NDArray, rho: float, *args, **kwargs):
        """Solve the reciprocal OZ equation for $e(q)$ in terms of $c(q)$."""
        return self.oz_solution_hq_from_cq(cq, rho, *args, **kwargs) - cq

    def oz_solution_eq_from_hq(self, hq: NDArray, rho: float, *args, **kwargs):
        """Solve the reciprocal OZ equation for $e(q)$ in terms of $h(q)$."""
        return hq - self.oz_solution_cq_from_hq(hq, rho)

    def oz_solution_hr_from_cr(self, cr: NDArray, *args, **kwargs):
        """Solve the OZ equation for $h(r)$ in terms of $c(r)$"""
        cq = self.grid.fourier_bessel_forward(cr)
        hq = self.oz_solution_hq_from_cq(cq, *args, **kwargs)
        return self.grid.fourier_bessel_backward(hq)

    def oz_solution_er_from_cr(self, cr: NDArray, *args, **kwargs):
        """Solve the OZ equation for $e(r) = h(r) - c(r)$"""
        cq = self.grid.fourier_bessel_forward(cr)
        eq = self.oz_solution_eq_from_cq(cq, *args, **kwargs)
        return self.grid.fourier_bessel_backward(eq)

    def oz_solution_er_from_hr(self, hr: NDArray, *args, **kwargs):
        """Solve the OZ equation for $e(r) = h(r) - c(r)$"""
        hq = self.grid.fourier_bessel_forward(hr)
        eq = self.oz_solution_eq_from_hq(hq, *args, **kwargs)
        return self.grid.fourier_bessel_backward(eq)

    def solve(self, potential: Type[potentials.Potential],
              rho: float, T: float=1.,
              e_init: NDArray=None,
              monitor: bool=False,
              restart: bool=False):
        """Solve HNC for a given potential, with an optional initial guess at cr.

        To iteratively solve the OZ equation we use the limited memory
        quasi-Newton method of:
        K.-C. Ng, J. Chem. Phys. 61, 2680 (1974).

        Args:
            potential: pair potential.
            rho: density.
            T: temperature (defaults to 1 so potential in units of kT).
            e_init: initial guess for e(r).
            monitor: if True will print iteration updates monitoring progress.
            restart: if True, will start afresh regardless. If False, will
                        start from the most recently converged solution (if
                        available) or cr_init if given.
        """

        phi = potential(self.r) / T
        if e_init is not None:
            assert not restart
            e = e_init.copy()
        elif not self.warmed_up or restart:
            h = np.zeros_like(self.r)
            b = np.zeros_like(self.r)
            e = self.oz_solution_er_from_hr(h, rho)
        else:
            h = self.h.copy()
            b = self.b.copy()
            e = self.e.copy()
        if np.any(np.isnan(e)): raise ValueError

        inner = lambda u, v: simpson(u*v, dx=self.grid.deltar)
        magnitude = lambda u: inner(u, u)**0.5

        # Memory of iterations for inferring hessian
        f = deque(maxlen=self.history_size) # input value of e
        g = deque(maxlen=self.history_size) # output value of e
        d = deque(maxlen=self.history_size) # change g[i] - f[i]

        f += [e]
        c = np.exp(-phi + f[-1] + b) - f[-1] - 1
        g += [self.oz_solution_er_from_cr(c, rho)]
        d += [g[-1] - f[-1]]
        prev_change = magnitude(d[-1])

        assert np.all(f[-1] > -1)
        assert not np.any(np.isnan(f[-1]))
        assert not np.any(np.isnan(g[-1]))

        for iter in range(self.niters):

            converged = prev_change < self.tol
            if converged:
                if iter > 0 and monitor:
                    print(f'{iter:>4}  {prev_change:<10.4g}')
                break

            if iter == 0 and monitor: print('iter  error')

            # Determine optimal direction for line search.
            if iter < self.npicard:
                # First few iterations we do not have curvature information
                # so we must resort to normal downhill direction (Picard). 
                step = d[-1]
                step_size = self.alpha

            else:
                # Switch to limited-memory quasi-Newton method of
                # K.-C. Ng, J. Chem. Phys. 61, 2680 (1974).

                delta = [d[-1] - d[-i-1] for i in range(1, len(f))]
                residual = [inner(d[-1], delta[i]) for i in range(len(f)-1)]

                A = np.empty((len(f)-1, len(f)-1))
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        A[i,j] = inner(delta[i], delta[j])
                        A[j,i] = A[i,j]

                α = np.zeros(len(f))
                α[1:] = np.linalg.solve(A, residual)
                α[0] = 1 - np.sum(α[1:])
                α = np.flipud(α)
                step = α.dot(g) - f[-1]
                step_size = 1.

            if np.any(np.isnan(step)): raise ValueError

            # Backtracking line search
            for line_iter in range(self.nline_searches):
                trial = f[-1] + step_size * step

                new_change = np.inf
                try:
                    b = self.bridge_closure(trial)
                    c = np.exp(-phi + trial + b) - trial - 1
                    e = self.oz_solution_er_from_cr(c, rho)
                    if np.any(np.isnan(e)): raise ValueError

                    delta = e - trial
                    new_change = magnitude(delta)

                except NotImplementedError as err:
                    raise err from None

                except: pass

                if np.isfinite(new_change): break
                step_size *= self.line_search_decay

            else:
                raise RuntimeError('line search stalled!')

            f += [trial]
            g += [e]
            d += [delta]
            prev_change = new_change

            if monitor and (iter % self.nmonitor == 0):
                print(f'{iter:>4}  {prev_change:<10.4g}')

        self.e = f[-1]
        self.b = b
        self.c = c
        self.potential = potential
        self.rho = rho
        self.T = T
        self.error = prev_change

        if converged:
            # Final check for finite size effects - the domain should be large
            # enough that $h(r) \to 0$  as $r \to \infty$
            nend = 10
            error = simpson(np.abs(self.h[-nend:]), self.r[-nend:])
            error /= simpson(np.ones_like(self.r[-nend:]), self.r[-nend:])
            converged = error < self.tol
            if converged: self.warmed_up = True

        self.converged = converged

        if monitor:
            if self.converged:
                if iter > 0: print(f'{self.name}: converged')
            else:
                print(f'{self.name}: iteration {iter:3d}, error = {self.error:0.3e}')
                print(f'{self.name}: failed to converge')

        return self.copy() # the user can name this 'soln' or something

    @property
    def pressure(self):
        """$\beta p$ via virial route."""
        assert self.converged
        f = self.potential.force(self.r)
        f[f > 1e4] = 0.
        I = simpson(self.r**3*self.g*f, self.r)
        return self.rho + 2/3 * np.pi * self.rho**2 / self.T * I

    @property
    def excess_chemical_potential(self):
        raise NotImplementedError


class PercusYevickSolver(OrnsteinZernikeSolver):
    def bridge_closure(self, e: NDArray, *args, **kwargs):
        """Closure to the OZ equation for $b(r)$."""
        with np.errstate(invalid='ignore'):
            return np.log(1 + e) - e


class HypernettedChainSolver(OrnsteinZernikeSolver):
    def bridge_closure(self, e: NDArray, *args, **kwargs):
        """Closure to the OZ equation for $b(r)$."""
        return np.zeros_like(e)

    @property
    def excess_chemical_potential(self):
        r"""Chemical potential/test particle route for $\beta \mu^\mathrm{ex}$."""
        assert self.converged
        I = simpson(self.r**2*(self.h*self.e/2 - self.c), self.r)
        return 4*np.pi * self.rho / self.T * I

    @property
    def excess_free_energy_density(self):
        r"""Free energy route for $\beta f^\mathrm{ex} = \beta F^\mathrm{ex} / V$."""
        return self.rho * (1 + self.excess_chemical_potential) - self.pressure


# Default to HNC closure
Solver = HypernettedChainSolver


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
    """Subclass for infinitely dilute solute inside solvent."""

    def __init__(self, solvent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solvent = solvent.copy()

    def init_kwargs(self):
        kwargs = super().init_kwargs()
        kwargs['solvent'] = self.solvent
        return kwargs

    def oz_solution_hq_from_cq(self, cq: NDArray, rho: float):
        """Solve the modified OZ equation for h, in reciprocal space."""
        return self.solvent.Sq * cq

    def solve(self, potential, *args, **kwargs):
        # rho = 0.0 as it is not needed
        return super().solve(potential, 0.0, *args, **kwargs)

# Below, cases added by Josh

class TestParticleRPA(Solver):
    """Subclass for mean-field DFT approach."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def oz_solution_hq_from_cq(self, cq: NDArray, rho: float):
        """Solution to the OZ equation in reciprocal space."""
        return cq / (1 + rho*self.vq) # force RPA closure in reciprocal term

    def solve(self, potential,
              rho: float, T: float=1.,
              *args, **kwargs):
        vr = potential(self.r) / T
        self.vq = self.grid.fourier_bessel_forward(vr) # forward transform v(r) to v(q)
        return super().solve(potential, rho, T, *args, **kwargs)

class SoluteTestParticleRPA(SoluteSolver):

    def __init__(self, *args, npicard=np.inf, **kwargs):
        try: super().__init__(*args, npicard=npicard, **kwargs)
        except TypeError:
            # If npicard given as positional argument drop it.
            super().__init__(*args, **kwargs)

    def oz_solution_hq_from_cq(self, cq: NDArray, rho: float):
        """Solution to the OZ equation in reciprocal space."""
        return cq - self.solvent.rho * self.solvent.hq * self.vq01 # RPA closure

    def solve(self, potential, T: float=1.,
              *args, **kwargs):
        vr01 = potential(self.r) / T
        self.vq01 = self.grid.fourier_bessel_forward(vr01) # forward transform v(r) to v(q)
        return super().solve(potential, T, *args, **kwargs)
