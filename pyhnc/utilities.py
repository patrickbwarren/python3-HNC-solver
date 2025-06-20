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

import argparse
import numpy as np

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

def add_solver_args(parser, alpha=0.2, niters=500, tol=1e-12):
    '''Add generic solver args to parser'''
    parser.add_argument('--alpha', default=alpha, type=float, help=f'Picard mixing fraction, default {alpha}')
    parser.add_argument('--niters', default=niters, type=int, help=f'max number of iterations, default {niters}')
    parser.add_argument('--tol', default=tol, type=float, help=f'tolerance for convergence, default {tol}')

def solver_args(args):
    '''Return a dict of generic solver args that can be used as **solver_args()'''
    return {'alpha': args.alpha, 'tol': args.tol, 'niters': args.niters}

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
    return dx * np.pad(0.5*(y[1:]+y[:-1]), (1, 0)) # pad with zero at start

def trapz(y, dx=1):
    '''Return the trapezium rule integral, drop-in replacement for np.trapz'''
    return trapz_integrand(y, dx=dx).sum()
