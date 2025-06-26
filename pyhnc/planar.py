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

import sys

import numpy as np, matplotlib.pyplot as plt

from scipy.integrate import quad, simpson, cumulative_simpson
from scipy.interpolate import interp1d

import pyfftw

from typing import Type
from collections import deque

from abc import ABC, abstractmethod
from numpy.typing import NDArray

try: from . import pyhnc, potentials
except ImportError: import pyhnc, potentials


class Wall(potentials.Potential):
    @property
    def nspecies(self):
        return 1

    @abstractmethod
    def potential(self, x: NDArray | float):
        raise NotImplementedError

    @abstractmethod
    def force(self, x: NDArray | float):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.potential(*args, **kwargs)


class PlanarGrid:

    def __init__(self, x: NDArray):

        self.x = x
        self.N = x.size
        self.full_size = 2 * self.N - 1
        self.freq_size = self.full_size // 2 + 1

        dx = np.diff(x)
        assert np.all(np.isclose(dx, dx[0]))
        self.dx = dx[0]
        self.L = x[-1] - x[0]

        self.q = 2*np.pi * np.fft.rfftfreq(self.full_size, d=self.dx)
        dq = np.diff(self.q)
        assert np.all(np.isclose(dq, dq[0]))
        self.dq = dq[0]
        assert np.isclose(self.dq, 2*np.pi/(self.dx * self.full_size))

        self.zero_phase = np.exp(-1j * self.q * self.x[0])

        self.a_padded = pyfftw.zeros_aligned(self.full_size, dtype='float64')
        self.b_padded = pyfftw.zeros_aligned(self.full_size, dtype='float64')
        self.fft_a_out = pyfftw.empty_aligned(self.freq_size, dtype='complex128')
        self.fft_b_out = pyfftw.empty_aligned(self.freq_size, dtype='complex128')
        self.ifft_out = pyfftw.empty_aligned(self.full_size, dtype='float64')

        flags = ('FFTW_ESTIMATE',)
        self.fftw_a = pyfftw.FFTW(self.a_padded, self.fft_a_out,
                                 direction='FFTW_FORWARD', flags=flags)
        self.fftw_b = pyfftw.FFTW(self.b_padded, self.fft_b_out,
                                 direction='FFTW_FORWARD', flags=flags)
        self.ifftw = pyfftw.FFTW(self.fft_a_out, self.ifft_out,
                                direction='FFTW_BACKWARD', flags=flags)

        # Indices for 'same' slicing in convolution
        self.start = (self.full_size - self.N) // 2
        self.stop = self.start + self.N


    def fourier_forward(self, fx: NDArray):
        """Forward transform of f(x) to reciprocal space."""
        assert fx.size == self.N
        self.a_padded[:self.N] = fx
        self.fftw_a()
        return self.fft_a_out.copy() * self.dx * self.zero_phase

    def fourier_backward(self, fq: NDArray):
        """Backward transform of f(q) to real space."""
        assert fq.size == self.freq_size
        self.fft_a_out[:] = fq / self.zero_phase
        self.ifftw()
        return self.ifft_out.copy() / self.dx

    def convolve(self, ax: NDArray, bx: NDArray):
        """Real space convolution of a(x) and b(x)."""

        assert ax.size == self.N
        assert bx.size == self.N

        self.a_padded[:self.N] = ax
        self.b_padded[:self.N] = bx

        self.fftw_a()
        self.fftw_b()
        self.fft_a_out[:] *= self.fft_b_out
        self.ifftw()

        return self.ifft_out[self.start:self.stop].real.copy() * self.dx


def test_fft(mu=0., sigma=2, N=1000, eps=1e-10):
    """Verify FFT and inverse on Gaussian where the analytical
    results are known."""
    x = np.linspace(-100, 100, N)
    grid = PlanarGrid(x)
    f = np.exp(-(x-mu)**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    fq = grid.fourier_forward(f)

    exact = np.exp(-1j * grid.q * mu) * np.exp(-0.5*(grid.q*sigma)**2)
    assert np.all(np.abs(fq - exact) < eps)

    f2 = grid.fourier_backward(fq)[:grid.N]
    assert np.all(np.isclose(f, f2))

def test_convolve(N=1000, eps=1e-10):
    """Check output is the same as a real-space convolution."""
    x = np.linspace(-100, 100, N)
    grid = PlanarGrid(x)
    a = np.random.random(x.size)
    b = np.random.random(x.size)
    # mu, sigma = 0., 2
    # b = np.exp(-(x-mu)**2/(2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    exact = np.convolve(a, b, 'same') * grid.dx

    aq = grid.fourier_forward(a)
    bq = grid.fourier_forward(b)
    fft_full = grid.fourier_backward(aq * bq / grid.zero_phase)
    fft_result1 = fft_full[grid.start:grid.stop]
    fft_result2 = grid.convolve(a, b)

    assert np.all(np.abs(exact - fft_result2) < eps)
    assert np.all(np.abs(fft_result1 - fft_result2) < eps)
    assert np.all(np.abs(exact - fft_result1) < eps)

if __name__ == '__main__':
    test_fft()
    test_convolve()
    print('all tests passed successfully')


class PlanarSolver(pyhnc.SoluteSolver):

    def __init__(self, solvent, *args, grid=None, cpar=None, **kwargs):

        if grid is None:
            rmax = solvent.grid.r[-1]
            N = solvent.grid.r.size
            x = np.linspace(-rmax, rmax, N)
            assert x[N//2] == 0.
            # dx = x[1] - x[0]
            grid = PlanarGrid(x)

        super().__init__(solvent, grid, *args, **kwargs)

        if cpar is not None:
            self.cpar = cpar.copy()
        else:
            # Pre-calculate $c_\parallel(x)$.
            cpar_sample = np.zeros_like(self.solvent.r)
            c = interp1d(self.solvent.r, self.solvent.c,
                        kind='cubic', fill_value='extrapolate')
            for i, z in enumerate(self.solvent.r[:-1]):
                smax = np.sqrt(self.solvent.r[-1]**2 - z**2)
                result = quad(lambda s: 2*np.pi * s * c(s), np.abs(z), smax)[0]
                cpar_sample[i] = result

            # Should be symmetric about x=0:
            cpar_sample = np.concatenate([np.flipud(cpar_sample[1:]), cpar_sample])
            x_sample = np.concatenate([-np.flipud(self.solvent.r[1:]), self.solvent.r])
            # Resample at more convenient points
            self.cpar = interp1d(x_sample, cpar_sample, kind='cubic')(self.x)

        self.converged = False

    def init_kwargs(self):
        kwargs = super().init_kwargs()
        kwargs['cpar'] = self.cpar
        return kwargs

    @property
    def x(self):
        return self.grid.x

    @property
    def r(self):
        """Alias to pass x coordinate to underlying SoluteSolver code which
        works with radial coordinates."""
        return self.x

    @property
    def dx(self):
        return self.grid.dx

    @property
    def dr(self):
        return self.dx

    @property
    def density(self):
        """Density of bulk system (far away from wall)."""
        return self.solvent.density

    @property
    def pressure(self):
        """Normal bulk pressure."""
        return self.solvent.pressure

    @property
    def density_profile(self):
        """Inhomogeneous density profile $\rho(x)$ around wall."""
        return self.density * (1 + self.h)

    @property
    def eq(self):
        return self.grid.fourier_forward(self.e)

    @property
    def bq(self):
        return self.grid.fourier_forward(self.b)

    @property
    def cq(self):
        return self.grid.fourier_forward(self.c)

    @property
    def hq(self):
        return self.grid.fourier_forward(self.h)

    @property
    def gq(self):
        return self.grid.fourier_forward(self.g)

    @property
    def excluded_region(self):
        """Volume per unit area of excluded region (left of the Gibbs dividing
        surface). For convenience we take the dividing surface to be at
        $x=0$."""
        return -np.min(self.x)

    @property
    def excess_chemical_potential(self):
        r"""Chemical potential/test particle route for $\beta \mu^\mathrm{ex}$."""
        assert self.converged
        integrand = self.h*(self.h-self.c)/2 - self.c

        # Subtract homogeneous value taking Gibbs surface at x=0.
        index = len(self.x)//4 # Point furthest from interface on either side
        pressure = integrand[index]
        hom = pressure * np.ones_like(integrand)
        hom[self.x>0] = 0.
        integrand -= hom

        # Evaluate integral, ignoring any numerical artefacts near the
        # left-most edge.
        I = simpson(integrand[index:], self.x[index:])
        return self.density * I

    @property
    def surface_tension(self):
        r"""Surface tension at the wall $\gamma = \Omega^{ex} / A$. This
        uses the chemical potential/test particle route with HNC closure for
        the change in grand potential $\Delta \Omega$. The homogeneous value
        must also be subtracted to obtain $\Omega^{ex}$, which requires a
        choice of the Gibbs dividing surface."""
        return self.excess_chemical_potential

    def oz_solution_cq_from_hq(self, hq: NDArray, *args, **kwargs):
        raise NotImplementedError

    def oz_solution_eq_from_cq(self, cq: NDArray, *args, **kwargs):
        raise NotImplementedError

    def oz_solution_eq_from_hq(self, hq: NDArray, *args, **kwargs):
        raise NotImplementedError

    def oz_solution_hq_from_cq(self, cq: NDArray, *args, **kwargs):
        """Solve the reciprocal OZ equation for $h(q)$ in terms of $c(q)$."""
        cparq = self.grid.fourier_forward(self.cpar)
        return cq / (1 - self.density*cparq)

    def oz_solution_h_from_c(self, c: NDArray, *args, **kwargs):
        """Solve the OZ equation for $h(x)$ in terms of $c(x)$"""
        cq = self.grid.fourier_forward(c)
        hq = self.oz_solution_hq_from_cq(cq, *args, **kwargs)
        h = self.grid.fourier_backward(hq / self.grid.zero_phase)
        return h[self.grid.start:self.grid.stop].real.copy()

    def oz_solution_e_from_c(self, c: NDArray, *args, **kwargs):
        """Solve the OZ equation for $e(x) = h(x) - c(x)$ in terms of $c(x)$"""
        h = self.oz_solution_h_from_c(c, *args, **kwargs)
        return h - c

    def oz_solution_e_from_h(self, h: NDArray, *args, **kwargs):
        """Solve the OZ equation for the indirect correlation function
        $e(x) = h(x) - c(x)$ in real space.
        """
        return self.density * self.grid.convolve(h, self.cpar)

    def inner_product(self, u, v):
        L = simpson(np.ones_like(self.x), self.x)
        return simpson(u*v, dx=self.dx) / L

    def optimal_direction(self, f, g, d):
        """Algorithm will try to switch to limited-memory quasi-Newton method
        of K.-C. Ng, J. Chem. Phys. 61, 2680 (1974) to speed up convergence.
        """
        # return self.picard_direction(f, g, d)
        delta = [d[-1] - d[-i-1] for i in range(1, len(f))]
        residual = [self.inner_product(d[-1], delta[i]) for i in range(len(f)-1)]

        A = np.empty((len(f)-1, len(f)-1))
        for i in range(len(A)):
            for j in range(i, len(A)):
                A[i,j] = self.inner_product(delta[i], delta[j])
                A[j,i] = A[i,j]

        if np.linalg.cond(A) > 1:
            # Revert to Picard if matrix singular
            return self.picard_direction(f, g, d)

        α = np.zeros(len(f))
        α[1:] = np.linalg.solve(A, residual)
        α[0] = 1 - np.sum(α[1:])
        α = np.flipud(α)
        step = α.dot(g) - f[-1]
        step_size = 1.

        return step, step_size

    def solve(self, *args, method: str='h', **kwargs):
        """Change default method to total correlation $h(r)$ as there are
        convergence issues (slow or non-existent) with indirect correlation
        function with planar case for some reason."""
        return super().solve(*args, **kwargs, method=method)
