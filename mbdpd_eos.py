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

# Generate EoS data for many-body DPD defined in PRE 68, 066702 (2003).
# In this model φ(r) = A(1−r)²/2 for r < 1 ; u(ρ) = πB(R²)²ρ²/30 ,
# and the weight function w(r) = 15/(2πR³) (1−r/R)² for r < R.

# This version is suitable for the map/reduce wrapper paradigm.

import os
import sys
import pyHNC
import argparse
import subprocess
import numpy as np

from numpy import pi as π
from numpy import cos, sin
from pyHNC import truncate_to_zero, ExtendedArgumentParser

parser = ExtendedArgumentParser(description='DPD EoS calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser, npicard=10000) # boost the possible number of Picard iteration steps
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('--header', default=None, help='set the name of the output files, default None')
parser.add_argument('--process', default=None, type=int, help='process number, default None')
parser.add_argument('-A', '--A', default=-40.0, type=float, help='repulsion amplitude')
parser.add_argument('-B', '--B', default=40.0, type=float, help='repulsion amplitude')
parser.add_argument('-R', '--R', default=0.75, type=float, help='repulsion r_c')
parser.add_argument('--rho', default='6.5', help='density or density range, default 6.5')
parser.add_argument('--rhobar', default=5.0, type=float, help='initial mean local density, default 5.0')
parser.add_argument('--drhobar', default=0.05, type=float, help='decrement looking for self consistency, default 0.05')
parser.add_bool_arg('--uprime', default=False, help='use du/dρ rather than u/ρ for MB potential')
parser.add_bool_arg('--rhoav', default=False, help='use ρ(r) rather than <ρ> in the MB potential')
parser.add_argument('--nrhoav', default=20, type=int, help='number of steps to converge, default 20')
parser.add_bool_arg('--refine', default=True, help='refine end point using interval halving')
parser.add_argument('--nrefine', default=20, type=int, help='number of interval halving steps, default 20')
parser.add_bool_arg('--condor', short_opt='-j', default=False, help='create a condor job')
parser.add_bool_arg('--reduce', short_opt='-r', default=True, help='create a DAGMan job to run the condor job')
parser.add_bool_arg('--clean', short_opt='-c', default=True, help='clean up intermediate files')
parser.add_bool_arg('--run', short_opt='-x', default=False, help='run the condor or DAGMan job')
parser.add_argument('--rmax', default=3.0, type=float, help='maximum in r for plotting, default 3.0')
parser.add_bool_arg('--show', default=False, help='show plots of things')
args = parser.parse_args()

args.script = os.path.basename(__file__)
args.executable = sys.executable

A, B, R = args.A, args.B, args.R

ρ_vals = pyHNC.as_linspace(args.rho)

opts = [f'--A={args.A}', f'--B={args.B}', f'--R={args.R}',
        f'--rho={args.rho}', 
        f'--rhobar={args.rhobar}', f'--drhobar={args.drhobar}',
        '--uprime' if args.uprime else '--no-uprime',
        '--rhoav' if args.rhoav else '--no-rhoav',
        f'--nrhoav={args.nrhoav}',
        '--refine' if args.refine else '--no-refine',
        f'--nrefine={args.nrefine}']

if args.condor: # create scripts to run jobs then exit

    njobs = len(ρ_vals)

    condor_job = f'{args.header}__condor.job'
    lines = ['should_transfer_files = YES',
             'when_to_transfer_output = ON_EXIT',
             'notification = never',
             'universe = vanilla',
             f'opts = ' + ' '.join(opts),
             f'transfer_input_files = pyHNC.py,{args.script}',
             f'executable = {args.executable}',
             f'arguments = {args.script} --header={args.header} $(opts) --process=$(Process)',
             f'output = {args.header}__$(Process).out',
             f'error = {args.header}__$(Process).err',
             f'queue {njobs}']
    with open(condor_job, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    run_command = f'condor_submit {condor_job}'

    if args.reduce:

        post_script = f'{args.header}__script.sh'
        with open(post_script, 'w') as f:
            f.write(f'cat {args.header}__*.dat | sort -g -k1 > {args.header}.dat\n')
            if args.clean:
                for ext in ['out', 'err', 'dat']:
                    f.write(f'rm -f {args.header}__*.{ext}\n')

        dag_job = f'{args.header}__dag.job'
        with open(dag_job, 'w') as f:
            f.write(f'JOB A {condor_job}\n')
            f.write(f'SCRIPT POST A /usr/bin/bash {post_script}\n')

        run_command = f'condor_submit_dag -notification Never {dag_job}'

    if args.run: # run if required, else print out the required command
        subprocess.call(run_command, shell=True)
    else:
        print(run_command)

    exit(0)

# *** Main computation starts ***

# In map/reduce mode, fish out the required density and coupling
# parameter using the value of process.

k = args.process if args.process is not None else 0

ρ = ρ_vals[k]

print(f'{args.script}: solving for A, B, R, ρ = {A}, {B}, {R}, {ρ}')

grid = pyHNC.Grid(**pyHNC.grid_args(args)) # make the initial working grid
r, Δr, q, Δq = grid.r, grid.deltar, grid.q, grid.deltaq # extract for use below
rbyR, qR = r/R, q*R # some reduced variables

solver = pyHNC.PicardHNC(grid, **pyHNC.solver_args(args))

# DPD potential and force law f = −dφ/dr.
# The array sizes here are ng-1, same as r[:].

φ = A/2 * truncate_to_zero((1-r)**2, r, 1)
φf = A * truncate_to_zero((1-r), r, 1)

# The many-body weight function (normalised) and its Fourier
# transform, and the derivative (unnormalised).

wr = 15/(2*π*R**3) * truncate_to_zero((1-rbyR)**2, r, R)
wq = 60*(2*qR + qR*cos(qR) - 3*sin(qR)) / qR**5
wf = truncate_to_zero((1-rbyR), r, R) # omit the normalisation

# Combine a search descent from initial value of rhobar with
# refinement using interval halving if requested.

i, ρbar_in, bracketed = 0, args.rhobar+args.drhobar, False

while not bracketed or (args.refine and i < args.nrefine):
    i = i + 1 # keep track of the number of cycles here
    ρbar_in = 0.5*(ρ1 + ρ2) if bracketed else (ρbar_in-args.drhobar)
    if args.uprime:
        v = φ + π*B*R**4/15 * 2*ρbar_in * wr
    else:
        v = φ + π*B*R**4/30 * 2*ρbar_in * wr
    soln = solver.solve(v, ρ)
    if soln.converged:
        hr = soln.hr
        f = φf + B * 2*ρbar_in * wf # MB DPD force law
        p = ρ + (A + 2*B*ρbar_in*R**4)*π*ρ**2/30 + 2/3*π*ρ**2*np.trapz(r**3*f*hr, dx=Δr)
        ρbar_out = ρ*(1 + 4*π*np.trapz(r**2*wr*hr, dx=Δr))
        if bracketed:
            print(f'{args.script}: bracketed {i:3d}:',
                  'ρ (ρ1, ρ2), ρbar_in, ρbar_out-ρbar_in, p = %f (%f, %f) %f %f %f' %
                  (ρ, ρ1, ρ2,  ρbar_in, ρbar_out-ρbar_in, p))
        else:
            print(f'{args.script}: descending {i:3d}:',
                  'ρ, ρbar_in, ρbar_out, p = %f %f %f %f' %
                  (ρ, ρbar_in, ρbar_out, p))
        if ρbar_out > ρbar_in: # jump to bracketed search
            if not bracketed: # reset the counter here
                i, bracketed = 0, True
            ρ1 = ρbar_in
        else:
            ρ2 = ρbar_in
    else: # HNC solver not converged
        print(f'{args.script}: not converged',
              '(bracketed):' if bracketed else '(descending):'
              'ρ, ρbar_in = %f %f' % (ρ, ρbar_in))
    if bracketed: # refine the interval
        ρ1, ρ2 = (ρbar_in, ρ2) if ρbar_out > ρbar_in else (ρ1, ρbar_in)

if args.rhoav: # attempt to improve the model by replacing ρbar with ρav(r)

    ρav = ρbar_out # a scalar, initially

    for i in range(args.nrhoav):

        v = φ + π*B*R**4/30 * 2*ρav * wr # generalised MB DPD potential
        soln = solver.solve(v, ρ)

        if not soln.converged:
            print(f'{args.script}: failed to converge')
            # solver.warmed_up = False

        # wgq = grid.fourier_bessel_forward(wr*gr)
        # wgXh = grid.fourier_bessel_backward(wgq*hq)
        # ρav = ρbar + ρ*wgXh
        # wρav = 4*π*np.trapz(r**2*wr*ρav, dx=Δr)

        hr, hq = soln.hr, soln.hq
        gr = 1.0 + hr # the pair function -- shouldn't be needed, with care!
        ρbar = ρ*(1 + 4*π*np.trapz(r**2*wr*hr, dx=Δr)) # <<< write as whq_zero = 4*π*np.trapz(r**2*w*h, dx=Δr)
        whq = grid.fourier_bessel_forward(wr*hr) # convolution
        wgXh = grid.fourier_bessel_backward((wq+whq)*hq) # convolution
        ρav = ρbar + ρ*wgXh # the corrected estimate # <<< rewrite first as  ρ*(1 + whq_zero ...)
        # in the next bit we should be able to separate out the '1' in the above
        wρav = 4*π*np.trapz(r**2*wr*ρav, dx=Δr) # weighted average, for tracking << check against above
        f = φf + B * 2*ρav * wf # generalised MB DPD force law
        # SHOULD BE ABLE TO SEPARATE THE MEAN-FIELD CONTRIBUTION HERE
        p = ρ + 2/3*π*ρ**2 * np.trapz(r**3*f*gr, dx=Δr) # can't separate out mean-field: ρbar is not constant
        print(f'{args.script}: (w + wh)*h {i:3d}:',
              'ρbar_in, ρbar, wρav, wρav/ρbar, p = %f\t%f\t%f\t%f\t%f' %
              (ρbar_in, ρbar, wρav, wρav/ρbar, p))

else: # if not args.rhoav

        ρav = ρbar_out # constant value here
        if args.uprime:
            v = φ + π*B*R**4/15 * 2*ρav * wr
        else:
            v = φ + π*B*R**4/30 * 2*ρav * wr
        soln = solver.solve(v, ρ)
        if not soln.converged:
            print(f'{args.script}: failed to converge at end')
        hr = soln.hr
        gr = 1.0 + hr # the pair function
        ρbar = ρ*(1 + 4*π*np.trapz(r**2*wr*hr, dx=Δr))
        f = φf + B * 2*ρbar * wf # generalised MB DPD force law
        p = ρ + (A + 2*B*ρbar*R**4)*π*ρ**2/30 + 2/3*π*ρ**2*np.trapz(r**3*f*hr, dx=Δr)
        ρav = ρbar * np.ones_like(r) # for plotting
        wρav = 4*π*np.trapz(r**2*wr*ρav, dx=Δr) # weighted average, ( = ρbar here one hopes)

print(f'{args.script}: FINAL: A, B, R, ρ = {A}, {B}, {R}, {ρ}',
      'ρbar_in, ρbar_out, ρbar, wρav, wρav/ρbar, p = %f\t%f\t%f\t%f\t%f\t%f' %
      (ρbar_in, ρbar_out, ρbar, wρav, wρav/ρbar, p))

if args.header: ### SORT THIS OUT SO THE LINE ORDER IS RIGHT AND A,B,R,rho printed with right # places

    data = {'A': A, 'B': B, 'R': R, 'rho': ρ, 'rhobar_in': ρbar_in, 'rhobar_out':ρbar_out,
            'rhobar':ρbar, 'wrhoav': wρav, 'wrhoavbyrhobar': wρav/ρbar, 'pressure': p}
    sub = '' if args.process is None else '__%d' % args.process # double underscore here
    data_file = f'{args.header}{sub}.dat' # only one data file
    with open(data_file, 'w') as f:
        if args.process is None or args.process == 0: # write for first file
            f.write(f'# {args.executable} {args.script} ' + ' '.join(opts) + '\n')
            f.write('## ' + '\t'.join([f'{key}({i+1})' for i, key in enumerate(data.keys())]) + '\n')
        f.write('\t'.join([('%g' % data[key]) for key in data]) + '\n')
    print(f'data in {data_file}')

if args.show:

    import matplotlib.pyplot as plt

    cut = r < args.rmax
    plt.plot(r[cut], gr[cut], 'g')        
    plt.plot(r[cut], v[cut]/10, 'r')
    plt.plot(r[cut], wr[cut]/(15/(2*π*R**3)), 'b')
    plt.plot(r[cut], ρav[cut]/ρbar, 'k')
    plt.xlabel('$r$')
    
    plt.show()
