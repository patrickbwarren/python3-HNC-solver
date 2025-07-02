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
from scipy.special import erf

class Potential(ABC):

    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new.__setstate__(self.__getstate__())
        return new

    @abstractmethod
    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    @property
    @abstractmethod
    def nspecies(self):
        raise NotImplementedError

    @abstractmethod
    def potential(self, r: NDArray | float):
        raise NotImplementedError

    @abstractmethod
    def force(self, r: NDArray | float):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.potential(*args, **kwargs)


class ShortRangeResidual(Potential):
    r"""Excess part of a potential after removing a long-range part.

    In solving the OZ equation it's often convenient to separate off the
    long-range tail using the asymptotic properties of correlation functions.
    In particular, $c(r) \to -\beta v(r)$ for large $r$. This can be exploited
    to use a smaller grid size and improve convergence. This class is a helper
    interface to access the remaining part of $v(r)$ after subtracting the
    long-range part that $v(r)$ approaches at large $r$.
    """

    def __getstate__(self):
        return {'full': self.full, 'long': self.long}

    def __repr__(self):
        return rf'<{type(self).__name__} full={self.full} long={self.long}>'

    def __init__(self, full, long):
        self.full = full
        self.long = long
        assert self.full.nspecies == self.long.nspecies

    @property
    def nspecies(self):
        assert self.full.nspecies == self.long.nspecies
        return self.full.nspecies

    def potential(self, r: float | NDArray):
        return self.full.potential(r) - self.long.potential(r)

    def force(self, r: float | NDArray):
        return self.full.force(r) - self.long.force(r)


class DPD(Potential):
    r"""Quadratic potential used for coarse-graining in dissipative particle
    dynamics (DPD):

        $$v(r) = \frac{A}{2} (1 - r/r_c)^2 \qquad \forall r \le \rcut\,,$$

    and $v(r) = 0$ for $r > r_c$. This potential is convenient as the force
    decreases linearly from $r = 0$ to $\rcut$ making it very soft and thus
    suitable for large time-steps.
    """

    def __getstate__(self):
        return {'A': self.A.copy(),
                'rcut': self.rcut.copy()}

    def __repr__(self):
        return rf'<{type(self).__name__} A={self.A.tolist()} rc={self.rcut.tolist()}>'

    def __init__(self, A: float | NDArray,
                 rcut: float | NDArray=1.):
        A = np.atleast_2d(A)
        assert A.shape[0] == A.shape[1]
        rcut = np.atleast_2d(rcut)
        assert rcut.shape[0] == rcut.shape[1]

        if 1 in rcut.shape:
            rcut = rcut[0][0] * np.ones_like(A)

        self.A = A
        self.rcut = rcut

    @property
    def nspecies(self):
        return len(self.A)

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        v = 0.5 * self.A[:,:,None] * (self.rcut[:,:,None] - r[None,None,:])**2
        v[r[None,None,:] > self.rcut[:,:,None]] = 0.
        v /= self.rcut[:,:,None]**2

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)

        f = self.A[:,:,None] * (self.rcut[:,:,None] - r[None,None,:])
        f[r[None,None,:] > self.rcut[:,:,None]] = 0.
        f /= self.rcut[:,:,None]**2

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


def test_dpd():
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert np.all(v.A == v2.A)
        assert np.all(v.rcut == v2.rcut)
        v2.A += 1
        v2.rcut += 1
        assert np.all(v.A != v2.A)
        assert np.all(v.rcut != v2.rcut)

    r = np.linspace(0, 10, 100)

    # Tests for single-component systems.

    A = 25
    v = DPD(A)
    assert np.isscalar(v.potential(1.))
    assert not np.isscalar(v.potential(r))
    assert v.copy().potential(1.) == v.potential(1.)
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    exact = np.array([-approx_fprime(rr, v.potential) for rr in r]).reshape(-1)
    assert np.allclose(v.force(r), exact, rtol=1e-6)

    rcut = 2.
    v = DPD(A, rcut)
    phi = v.potential(r)
    assert np.allclose(phi[r > rcut], 0.)
    assert np.all(~np.isclose(phi[r < rcut], 0.))

    # Test it also works for binary and ternary mixtures.
    for n in [2, 3]:
        A = 25 * np.ones((n, n))
        v = DPD(A)
        assert v.potential(1.).shape == A.shape
        assert v.potential(r).shape == (n, n, r.size)
        test_copy(v, v.copy())
        test_copy(v, pickle.loads(pickle.dumps(v)))


class GaussianIonLongRange(Potential):
    r"""Long range part of GaussianIon."""

    def __getstate__(self):
        return {'full': self.full}

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.full.z} α={self.full.α} lB={self.full.lB}>'

    def __init__(self, full):
        self.full = full

    @property
    def nspecies(self):
        return self.full.nspecies

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        with np.errstate(invalid='ignore'):
            v = self.full.lB * erf(self.full.α**0.5 * r) / r
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def potential_fourier(self, k: float | NDArray):
        k = np.atleast_1d(k)

        with np.errstate(invalid='ignore'):
            v = 4*np.pi * self.full.lB * np.exp(-k**2 / (4*self.full.α)) / k**2
        v = np.outer(self.full.z, self.full.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)
        α = self.full.α

        with np.errstate(invalid='ignore'):
            f = self.full.lB * (
                    erf(α**0.5 * r)/r - 2*(α/np.pi)**0.5 * np.exp(-α*r**2)
                ) / r
        f = np.outer(self.full.z, self.full.z)[:,:,None] * f[None,None,:]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


class GaussianIon(Potential):
    r"""Interactions between ions with normally distributed charges:

        $$\rho_i(r) = \ell_\mathrm{B} z_i
        \left( \frac{\alpha}{\pi} \right)^{3/2} e^{-\alpha r^2}\,,$$

    where $r$ is the distance from the atom centre, $\ell_\mathrm{B}$ is the
    Bjerrum length and $z_i$ is the valence of species $i$.
    """

    def __getstate__(self):
        return {'z': self.z.copy(),
                'α': self.α.copy(),
                'lB': self.lB}

    def __repr__(self):
        return rf'<{type(self).__name__} z={self.z} α={self.α} lB={self.lB}>'

    def __init__(self, z: float | NDArray, α: float, lB: float=1.):
        self.z = np.atleast_1d(z)
        self.α = np.array(α)
        assert self.α.size == 1
        self.lB = lB

        self.long = GaussianIonLongRange(self)
        self.short = ShortRangeResidual(self, self.long)

    @property
    def nspecies(self):
        return len(self.z)

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        with np.errstate(invalid='ignore'):
            v = self.lB * erf((0.5*self.α)**0.5 * r) / r
        v = np.outer(self.z, self.z)[:,:,None] * v[None,None,:]

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)
        α = self.α

        with np.errstate(invalid='ignore'):
            f = self.lB * (
                    erf((0.5*self.α)**0.5 * r)/r -
                    (2*self.α/np.pi)**0.5 * np.exp(-0.5*self.α*r**2)
                ) / r
        f = np.outer(self.z, self.z)[:,:,None] * f[None,None,:]

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


def test_gaussian_ion():
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert np.all(v.α == v2.α)
        assert np.all(v.z == v2.z)
        assert v.lB == v2.lB
        v2.α += 1
        v2.z *= 2
        v2.lB += 1
        assert np.all(v.α != v2.α)
        assert np.all(v.z != v2.z)
        assert v.lB != v2.lB

    v = GaussianIon([1, -1], 1)
    assert np.allclose(v.copy().potential(1.), v.potential(1.))
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    r = np.linspace(0, 10, 100)[1:]

    for i in range(v.nspecies):
        for j in range(v.nspecies):
            for vv in [v, v.short, v.long]:
                f = lambda r: vv.potential(r)[i,j]
                exact = np.array([-approx_fprime(rr, f) for rr in r]).reshape(-1)
                assert np.allclose(vv.force(r)[i,j], exact, rtol=1e-6)

    short, long = v.short.potential(r), v.long.potential(r)
    assert np.allclose(v.potential(r), short + long)


class LennardJones(Potential):
    r"""Standard truncated Lennard-Jones potential used in atomic simulations:

        $$v(r) = 4\epsilon \left( \left( \frac{\sigma}{r} \right)^12 - \left( \frac{\sigma}{r} \right)^6 \right)\,.$$

    If truncated at some rcut < infinity) then this becomes

        $$v_\text{truncate}(r) = v(r) - v(rcut)\,.$$
    """

    def __getstate__(self):
        return {'sigma': self.sigma,
                'epsilon': self.epsilon,
                'rcut': self.rcut,
                'vshift': self.vshift}

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

    @property
    def nspecies(self):
        return 1

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        r6inv = (self.sigma/r)**6
        v = 4*self.epsilon * (r6inv**2 - r6inv) - self.vshift
        v[r >= self.rcut] = 0.

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)

        rinv = self.sigma/r
        r6inv = rinv**6
        f = 4*self.epsilon * (12*r6inv**2 - 6*r6inv) / r
        f[r >= self.rcut] = 0.

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f


def test_lj():
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert np.all(v.sigma == v2.sigma)
        assert np.all(v.epsilon == v2.epsilon)
        assert np.all(v.rcut == v2.rcut)
        v2.sigma += 1
        v2.epsilon += 1
        v2.rcut += 1
        assert np.all(v.sigma != v2.sigma)
        assert np.all(v.epsilon != v2.epsilon)
        assert np.all(v.rcut != v2.rcut)

    r = np.linspace(1, 10, 100)

    # Tests for single-component systems.

    v = LennardJones()
    assert np.isscalar(v.potential(1.))
    assert not np.isscalar(v.potential(r))
    assert v.copy().potential(1.) == v.potential(1.)
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    exact = np.array([-approx_fprime(rr, v.potential) for rr in r]).reshape(-1)
    assert np.allclose(v.force(r), exact, rtol=1e-6)

    rcut = 2.
    v = LennardJones(rcut=rcut)
    phi = v.potential(r)
    assert np.allclose(phi[r > rcut], 0.)
    assert np.all(~np.isclose(phi[r < rcut], 0.))


class Gaussian(Potential):
    r"""A simple Gaussian potential. This is primarily used to test Ng
    splitting (cf. `GaussianSplit` and `pyhnc.OrnsteinZernikeSolver`)."""

    def __getstate__(self):
        return {'alpha': self.alpha.copy()}

    def __repr__(self):
        return rf'<Gaussian α={self.alpha.tolist()}>'

    def __init__(self, alpha: float | NDArray=1.):
        alpha = np.atleast_2d(alpha)
        assert alpha.shape[0] == alpha.shape[1]
        self.alpha = alpha

    @property
    def nspecies(self):
        return len(self.alpha)

    def potential(self, r: float | NDArray):
        r = np.atleast_1d(r)

        α = self.alpha[:,:,None]
        v = (α/np.pi)**1.5 * np.exp(-α * r**2)

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def potential_fourier(self, k: float | NDArray):
        k = np.atleast_1d(k)

        α = self.alpha[:,:,None]
        v = np.exp(-k**2 / (4*α))

        v = np.squeeze(v)
        if v.ndim == 0: v = v.item()
        return v

    def force(self, r: float | NDArray):
        r = np.atleast_1d(r)

        α = self.alpha[:,:,None]
        f = 2*r * (α/np.pi)**1.5 * np.exp(-α * r**2)

        f = np.squeeze(f)
        if f.ndim == 0: f = f.item()
        return f

class GaussianSplit(Gaussian):
    r"""Helper class so that whole Gaussian will be splintered off while
    solving the Ornstein-Zernike equation (cf. `pyhnc.OrnsteinZernikeSolver)
    to test Ng splitting."""

    def __init__(self, alpha: float | NDArray=1.):
        super().__init__(alpha)
        self.long = Gaussian(alpha)
        self.short = ShortRangeResidual(self, self.long)


def test_gaussian():
    import pickle
    def test_copy(v, v2):
        assert v2 is not v
        assert np.all(v.alpha == v2.alpha)
        v2.alpha += 1
        assert np.all(v.alpha != v2.alpha)

    r = np.linspace(1, 10, 100)

    # Tests for single-component systems.

    v = Gaussian()
    assert np.isscalar(v.potential(1.))
    assert not np.isscalar(v.potential(r))
    assert v.copy().potential(1.) == v.potential(1.)
    test_copy(v, v.copy())
    test_copy(v, pickle.loads(pickle.dumps(v)))

    from scipy.optimize import approx_fprime
    exact = np.array([-approx_fprime(rr, v.potential) for rr in r]).reshape(-1)
    assert np.allclose(v.force(r), exact, rtol=1e-6)



if __name__ == '__main__':
    test_dpd()
    test_gaussian_ion()
    test_lj()
    test_gaussian()
