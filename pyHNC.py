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

# Hyper-netted chain (HNC) solver for Ornstein-Zernike (OZ) equation.

import pyfftw
import numpy as np

def truncate_to_zero(v, r, rc):
    '''Utility function to truncate a potential'''
    v[r>rc] = 0.0
    return v

# Provide a grid as a working platform.  This is the pair of arrays
# r(:) and q(:) initialised to match the desired ng (# grid points)
# and Δr.  Note that the array lengths are actually ng-1.  A real odd
# discrete Fourier transform (RODFT00) is also initialised, with
# functions to do forward and backward Fourier-Bessel transforms
# (radial 3d). For implementation details of FFTW see fftw_test.py.
# In particular note that Δr×Δk = π / (n+1) where n is the length of
# the FFT arrays.  Testing indicate the best efficiency is obtained
# when ng = 2^r with r being an integer.

class Grid:

    def __init__(self, ng=8192, deltar=0.02, monitor=False):
        '''Initialise grids with the desired size and spacing'''
        self.version = '1.0' # for reporting purposes
        self.ng = ng
        self.deltar = deltar
        self.deltaq = np.pi / (self.deltar*self.ng) # as above
        self.r = self.deltar * np.arange(1, self.ng) # start from 1, and of length ng-1
        self.q = self.deltaq * np.arange(1, self.ng) # ditto
        self.fftwx = pyfftw.empty_aligned(self.ng-1)
        self.fftwy = pyfftw.empty_aligned(self.ng-1)
        self.fftw = pyfftw.FFTW(self.fftwx, self.fftwy, direction='FFTW_RODFT00', flags=('FFTW_ESTIMATE',))
        if monitor:
            print('Grid: ng, Δr, Δq =', self.ng, f'(2^{int(0.5+np.log(self.ng)/np.log(2.0))})', self.deltar, self.deltaq)
            print('FFTW: initialised, array sizes =', self.ng-1)

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

# What's being solved here is the Ornstein-Zernike (OZ) equation in
# the form h(q) = c(q) + ρ h(q) c(q) in combination with the HNC
# closure g(r) = exp[ - v(r) + h(r) - c(r)], using Picard iteration.
# Here c(r) is the direct correlation function, h(r) = g(r) - 1 is the
# total correlation function, and v(r) is the potential.  In practice
# the OZ equation and the HNC closure are written in terms of the
# indirect correlation function e(r) = h(r) - c(r).  An initial guess
# if the solver is not warmed up is c(r) = - v(r) (ie, the RPA soln).

class PicardHNC:

    def __init__(self, grid, alpha=0.2, tol=1e-12, max_iter=500, monitor=False):
        '''Initialise basic data structure'''
        self.grid = grid
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.monitor = monitor
        self.converged = False
        self.warmed_up = False
        if self.monitor:
            print('HNC: grid ng = %i, Δr = %g, Δq = %g' % (self.grid.ng, self.grid.deltar, self.grid.deltaq))
            print('HNC: α = %g, tol = %0.1e, max_picard = %i' % (self.alpha, self.tol, self.max_iter))

    def solve(self, vr, rho, cr_init=None):
        '''Solve HNC for a given potential, with an optional initial guess at cr'''
        cr = cr_init if cr_init else self.cr if self.warmed_up else -vr
        for i in range(self.max_iter):
            cq = self.grid.fourier_bessel_forward(cr) # forward transform c(r) to c(q)
            eq = cq / (1 - rho*cq) - cq # solve the OZ equation for e(q)
            er = self.grid.fourier_bessel_backward(eq) # back transform e(q) to e(r)
            cr_new = np.exp(-vr+er) - er - 1 # iterate with the HNC closure
            cr = self.alpha * cr_new + (1-self.alpha) * cr # apply a Picard mixing rule
            self.error = np.sqrt(np.trapz((cr_new - cr)**2, dx=self.grid.deltar)) # convergence test
            self.converged = self.error < self.tol
            if self.monitor and (i % 50 == 0 or self.converged):
                print('Picard: iteration %3i, error = %0.3e' % (i, self.error))
            if self.converged:
                break
        if self.converged: 
            self.cr = cr_new # use the most recent calculation
            self.cq = self.grid.fourier_bessel_forward(cr)
            self.hr = self.cr + er # total correlation function
            self.hq = self.cq + eq
            self.warmed_up = True
        else: # we leave it to the user to check if self.converged is False :-)
            pass
        if self.monitor:
            if self.converged:
                print('Picard: converged')
            else:
                print('Picard: iteration %3i, error = %0.3e' % (i, self.error))
                print('Picard: failed to converge')
        return self # the user can name this 'soln' or something

# Utility functions below here for setting arguments, pretty printing dataframes

def add_grid_args(parser):
    '''Add generic grid arguments to a parser'''
    parser.add_argument('-m', '--monitor', action='store_true', help='monitor convergence')
    parser.add_argument('--grid', action='store', default=None, help='grid definition using deltar or deltar/ng, eg 0.02 or 0.02/8192')
    parser.add_argument('--ngrid', action='store', default='2^13', help='number of grid points, default 2^13 = 8192')
    parser.add_argument('--deltar', action='store', default=0.02, type=float, help='grid spacing, default 0.02')

def grid_args(args):
    '''Return a dict of grid args, that can be used as **grid_args()'''
    if args.grid:
        if '/' in args.grid:
            args_deltar, args.ngrid = args.grid.split('/')
            args.deltar = float(args_deltar)
        else:
            args.deltar = float(args.grid)
            args.r = int(1+np.log(np.pi/(args.deltar**2))/np.log(2.0))
            args.ngrid = str(2**args.r)
    ng = eval(args.ngrid.replace('^', '**')) # catch 2^10 etc
    return {'ng':ng, 'deltar': args.deltar, 'monitor': args.monitor}

def add_solver_args(parser):
    '''Add generic solver args to parser (monitor is assigned already)'''
    parser.add_argument('--alpha', action='store', default=0.2, type=float, help='Picard mixing fraction, default 0.2')
    parser.add_argument('--picard', action='store', default=500, type=int, help='max number of Picard steps, default 500')
    parser.add_argument('--tol', action='store', default=1e-12, type=float, help='tolerance for convergence, default 1e-12')

def solver_args(args):
    '''Return a dict of generic solver args that can be used as **solver_args()'''
    return {'alpha': args.alpha, 'tol': args.tol, 'max_iter': args.picard, 'monitor': args.monitor}

# Make the data output suitable for plotting in xmgrace if captured by redirection
# stackoverflow.com/questions/30833409/python-deleting-the-first-2-lines-of-a-string

def df_to_agr(df):
    '''Convert a pandas DataFrame to a string for an xmgrace data set'''
    header_row = '# ' + '\t'.join([f'{col}({i+1})' for i, col in enumerate(df.columns)])
    data_rows = df.to_string(index=False).split('\n')[2:]
    return '\n'.join([header_row] + data_rows)

def as_linspace(as_range):
    '''Convert an Abramowitz and Stegun style range "start(step)end" to an np array'''
    start, step, end = [float(eval(x)) for x in as_range.replace('(', ':').replace(')', ':').split(':')]
    npt = int((end-start)/step + 1.5)
    return np.linspace(start, end, npt)
