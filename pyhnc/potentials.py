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

try: from .utilities import *
except ImportError: from utilities import *

from abc import ABC, abstractmethod
from typing import Type
from numpy.typing import NDArray


class Potential(ABC):
    @abstractmethod
    def potential(self, r: NDArray | float):
        raise NotImplementedError

    @abstractmethod
    def force(self, r: NDArray | float):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.potential(*args, **kwargs)


class DPD(Potential):
    r"""Quadratic potential used for coarse-graining in dissipative particle
    dynamics (DPD):

        $$v(r) = \frac{A}{2} (\sigma - r)^2 \qquad \forall r \le \sigma\,,$$

    and $v(r) = 0$ for $r > \sigma$. This potential is convenient as the force
    decreases linearly from $r = 0$ to $\sigma$ making it very soft and thus
    suitable for large time-steps.
    """

    def __init__(self, A: float | NDArray,
                 sigma: float | NDArray=1.):
        A = np.atleast_2d(A)
        assert A.shape[0] == A.shape[1]
        sigma = np.atleast_2d(sigma)
        assert sigma.shape[0] == sigma.shape[1]

        if 1 in sigma.shape:
            sigma = sigma[0][0] * np.ones_like(A)

        self.A = A
        self.sigma = sigma

    @property
    def nspecies(self):
        return len(self.A)

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        v = 0.5 * self.A[:,:,None] * (self.sigma[:,:,None] - r[None,None,:])**2
        v[r[None,None,:] > self.sigma[:,:,None]] = 0.
        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()

        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)

        f = -self.A[:,:,None] * (self.sigma[:,:,None] - r[None,None,:])
        f[r[None,None,:] > self.sigma[:,:,None]] = 0.
        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()

        return f


def test_dpd():
    r = np.linspace(0, 10, 100)

    # Tests for single-component systems.

    A = 25
    v = DPD(A)
    assert np.isscalar(v.potential(1.))
    assert not np.isscalar(v.potential(r))

    from scipy.optimize import approx_fprime
    exact = np.array([approx_fprime(rr, v.potential) for rr in r]).reshape(-1)
    assert np.allclose(v.force(r), exact, rtol=1e-6)

    sigma = 2.
    v = DPD(A, sigma)
    phi = v.potential(r)
    assert np.all(np.isclose(phi[r > sigma], 0.))
    assert np.all(~np.isclose(phi[r < sigma], 0.))

    # Test it also works for binary and ternary mixtures.
    for n in [2, 3]:
        A = 25 * np.ones((n, n))
        v = DPD(A)
        assert v.potential(1.).shape == A.shape
        assert v.potential(r).shape == (n, n, r.size)


class LennardJones(Potential):
    def __init__(self, sigma: float = 1.,
                 epsilon: float = 1.,
                 rcut: float = None):
        self.sigma = sigma
        self.epsilon = epsilon
        if rcut is None: rcut = 2.5*self.sigma
        self.rcut = rcut

        r6inv = (self.sigma/self.rcut)**6
        self.vshift = 4*self.epsilon * (r6inv**2 - r6inv)

    def __call__(self, *args, **kwargs):
        return self.potential(*args, **kwargs)

    def potential(self, r):
        r6inv = (self.sigma/r)**6
        v = 4*self.epsilon * (r6inv**2 - r6inv) - self.vshift
        v[r >= self.rcut] = 0.
        return v

    def force(self, r):
        rinv = self.sigma/r
        r6inv = rinv**6
        f = 4*self.epsilon * (12*r6inv**2 - 6*r6inv) / r
        f[r >= self.rcut] = 0.
        return f


if __name__ == '__main__':
    test_dpd()
