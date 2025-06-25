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
    def __init__(self, A: float, *args, sigma: float=1., **kwargs):
        self.A = A
        self.sigma = sigma

    def potential(self, r):
        return truncate_to_zero(0.5*self.A*(self.sigma-r)**2/self.sigma, r, 1)

    def force(self, r):
        return truncate_to_zero(self.A*(1-r/self.sigma), r, 1)


class LennardJones(Potential):
    def __init__(self, sigma: float = 1., epsilon: float = 1., rcut: float = None,
                 *args, **kwargs):
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
