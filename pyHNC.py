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
import argparse
import numpy as np

version = '1.0' # for reporting purposes

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

    def __init__(self, ng=8192, deltar=0.02):
        '''Initialise grids with the desired size and spacing'''
        self.ng = ng
        self.deltar = deltar
        self.deltaq = np.pi / (self.deltar*self.ng) # as above
        self.r = self.deltar * np.arange(1, self.ng) # start from 1, and of length ng-1
        self.q = self.deltaq * np.arange(1, self.ng) # ditto
        self.fftwx = pyfftw.empty_aligned(self.ng-1)
        self.fftwy = pyfftw.empty_aligned(self.ng-1)
        self.fftw = pyfftw.FFTW(self.fftwx, self.fftwy, direction='FFTW_RODFT00',
                                flags=('FFTW_ESTIMATE',))
        r = round(0.5+np.log(self.ng)/np.log(2.0)) # the exponent if ng = 2^r
        self.details = f'Grid: ng = {self.ng} = 2^{r}, Δr = {self.deltar}, ' \
            f'Δq = {self.deltaq:0.3g}, |FFTW arrays| = {self.ng-1}'

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

    def __init__(self, grid, alpha=0.2, tol=1e-12, npicard=500, nmonitor=50):
        '''Initialise basic data structure'''
        self.grid = grid
        self.alpha = alpha
        self.tol = tol
        self.npicard = npicard
        self.nmonitor = nmonitor
        self.converged = False
        self.warmed_up = False
        self.details = f'HNC: α = {self.alpha}, tol = {self.tol:0.1e}, npicard = {self.npicard}'

    def solve(self, vr, rho, cr_init=None, monitor=False):
        '''Solve HNC for a given potential, with an optional initial guess at cr'''
        cr = np.copy(cr_init) if cr_init is not None else np.copy(self.cr) if self.warmed_up else -np.copy(vr)
        for i in range(self.npicard):
            cq = self.grid.fourier_bessel_forward(cr) # forward transform c(r) to c(q)
            eq = cq / (1 - rho*cq) - cq # solve the OZ equation for e(q)
            er = self.grid.fourier_bessel_backward(eq) # back transform e(q) to e(r)
            cr_new = np.exp(-vr+er) - er - 1 # iterate with the HNC closure
            cr = self.alpha * cr_new + (1-self.alpha) * cr # apply a Picard mixing rule
            if any(np.isnan(cr)): # break early if something blows up
                break
            self.error = np.sqrt(np.trapz((cr_new - cr)**2, dx=self.grid.deltar)) # convergence test
            self.converged = self.error < self.tol
            if monitor and (i % self.nmonitor == 0 or self.converged):
                iter_s = f'pyHNC.solve: Picard iteration %{len(str(self.npicard))}d,' % i
                print(f'{iter_s} error = {self.error:0.3e}')
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
        if monitor:
            if self.converged:
                print('pyHNC.solve: Picard converged')
            else:
                print(f'pyHNC.solve: Picard iteration {i:3d}, error = {self.error:0.3e}')
                print('pyHNC.solve: Picard failed to converge')
        return self # the user can name this 'soln' or something

# Extend the ArgumentParser class to be able to add boolean options, adapted from
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

class ExtendedArgumentParser(argparse.ArgumentParser):

    def add_bool_arg(self, long_opt, short_opt=None, default=False, help=None):
        '''Add a mutually exclusive --opt, --no-opt group with optional short opt'''
        opt = long_opt.removeprefix('--')
        group = self.add_mutually_exclusive_group(required=False)
        help_string = None if not help else help if not default else f'{help} (default)'
        if short_opt:    
            group.add_argument(short_opt, f'--{opt}', dest=opt, action='store_true', help=help_string)
        else:
            group.add_argument(f'--{opt}', dest=opt, action='store_true', help=help_string)
        help_string = None if not help else f"don't {help}" if default else f"don't {help} (default)"        
        group.add_argument(f'--no-{opt}', dest=opt, action='store_false', help=help_string)
        self.set_defaults(**{opt:default})

# Utility functions below here for setting arguments, pretty printing dataframes, etc

def power_eval(ng):
    '''Evalue ng as a string, eg 2^10 -> 1024'''
    return eval(ng.replace('^', '**')) # catch 2^10 etc

def add_grid_args(parser, ngrid='2^13', deltar=0.02):
    '''Add generic grid arguments to a parser'''
    parser.add_argument('--grid', default=None, help='grid using deltar or deltar/ng, eg 0.02 or 0.02/8192')
    parser.add_argument('--ngrid', default=ngrid, help=f'number of grid points, default {ngrid} = {power_eval(ngrid)}')
    parser.add_argument('--deltar', default=deltar, type=float, help='grid spacing, default 0.02')

def grid_args(args):
    '''Return a dict of grid args, that can be used as **grid_args()'''
    if args.grid:
        if '/' in args.grid:
            args_deltar, args.ngrid = args.grid.split('/')
            args.deltar = float(args_deltar)
        else:
            args.deltar = float(args.grid)
            r = 1 + round(np.log(np.pi/(args.deltar**2))/np.log(2))
            args.ngrid = str(2**r)
    ng = power_eval(args.ngrid)
    return {'ng':ng, 'deltar': args.deltar}

def add_solver_args(parser, alpha=0.2, npicard=500, tol=1e-12):
    '''Add generic solver args to parser'''
    parser.add_argument('--alpha', default=alpha, type=float, help=f'Picard mixing fraction, default {alpha}')
    parser.add_argument('--npicard', default=npicard, type=int, help=f'max number of Picard steps, default {npicard}')
    parser.add_argument('--tol', default=tol, type=float, help=f'tolerance for convergence, default {tol}')

def solver_args(args):
    '''Return a dict of generic solver args that can be used as **solver_args()'''
    return {'alpha': args.alpha, 'tol': args.tol, 'npicard': args.npicard}

# Make the data output suitable for plotting in xmgrace if captured by redirection
# stackoverflow.com/questions/30833409/python-deleting-the-first-2-lines-of-a-string

def df_header(df):
    '''Generate a header of column names as a list'''
    return [f'{col}({i+1})' for i, col in enumerate(df.columns)]

def df_to_agr(df):
    '''Convert a pandas DataFrame to a string for an xmgrace data set'''
    header_row = '#  ' + '  '.join(df_header(df))
    data_rows = df.to_string(index=False).split('\n')[1:]
    return '\n'.join([header_row] + data_rows)

# Convert a variety of formats and return the corresponding NumPy array.
# Options can be Abramowitz and Stegun style 'start:step:end' or 'start(step)end',
# NumPy style 'start,end,npt'.  A single value is returned as a 1-element array.
# A pair of values separated by a comma is returned as a 2-element array.

def as_linspace(as_range):
    '''Convert a range expressed as a string to an np.linspace array'''
    if ',' in as_range:
        vals = as_range.split(',')
        if len(vals) == 2: # case start,end
            xarr = np.array([float(vals[0]), float(vals[1])])
        else:
            start, end, npt = float(vals[0]), float(vals[1]), int(vals[2])
            xarr = np.linspace(start, end, npt)
    elif ':' in as_range or '(' in as_range:
        vals = as_range.replace('(', ':').replace(')', ':').split(':')
        if len(vals) == 2: # case start:end for completeness
            xarr = np.array([float(vals[0]), float(vals[1])])
        else:
            start, step, end = float(vals[0]), float(vals[1]), float(vals[2])
            npt = int((end-start)/step + 1.5)
            xarr = np.linspace(start, end, npt)
    else:
        xarr = np.array([float(as_range)])
    return xarr

def truncate_to_zero(v, r, rc):
    '''Utility function to truncate a potential'''
    v[r>rc] = 0.0
    return v

def grid_spacing(x):
    '''Utility to return the grid space assuming the array is evenly spaced'''
    return x[1] - x[0]

def trapz_integrand(y, dx=1):
    '''Implement trapezium rule and return integrand'''
    return dx * np.pad((y[1:] + y[:-1]) / 2.0, (1, 0), 'constant') # pad with zero at start

def trapz(y, dx=1):
    '''Return the trapezium rule integral, drop-in replacement for np.trapz'''
    return trapz_integrand(y, dx=dx).sum()
