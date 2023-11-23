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

# Use the HNC package to solve nDPD potentials, optionally plotting
# the pair distribution function and potential.  The nDPD model is
# described in Sokhan et al., Soft Matter 19, 5824 (2023).

# Extracted from Fig 5 in this paper, at T* = T / T_c = 0.4 the
# liquidus points are ρ* = ρ / ρc = 4.15 (n = 3) and 3.56 (n = 4).
# These state points can be refined with:
# ./ndpd_liquidus.py -n 3 -T 0.4 -r 4.2,4.3 --> ρ = 4.26955
# ./ndpd_liquidus.py -n 4 -T 0.4 -r 3.5,3.6 --> ρ = 3.5734

import os
import sys
import pyHNC
import argparse
import subprocess
import numpy as np
import pyHNC

from numpy import pi as π
from pyHNC import truncate_to_zero, ExtendedArgumentParser

failure = None

def multiply_by(x, iff=True):
    '''utility: use to conditionally multiply something by x'''
    return x if iff else 1.0

parser = ExtendedArgumentParser(description='nDPD HNC calculator')
pyHNC.add_grid_args(parser)
pyHNC.add_solver_args(parser, alpha=0.01, npicard=20000) # greatly reduce alpha and increase npicard here !!
parser.add_argument('-v', '--verbose', action='count', help='more details (repeat as required)')
parser.add_argument('job_name', nargs='?', default=None, help='set the name of the output files, default None')
parser.add_argument('--process', default=None, type=int, help='process number, default None')
parser.add_argument('-n', '--n', default='3', help='governing exponent, default 2')
parser.add_argument('-A', '--A', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-B', '--B', default=None, type=float, help='overwrite repulsion amplitude, default none')
parser.add_argument('-T', '--T', default='1.0', help='temperature or temperature range, default 1.0')
parser.add_argument('-r', '--rho', default='4.2,4.3', help='bracketing density, default 4.0')
parser.add_argument('--ptol', default=1e-5, type=float, help='warn condition for vanishing pressure')
parser.add_argument('--np', default=10, type=int, help='max number of iterations')
parser.add_argument('--ns', default=20, type=int, help='max number of search steps')
parser.add_bool_arg('--cold', default=False, help='force a cold start every time')
parser.add_bool_arg('--relative', default=True, help='ρ, T relative to critical values')
parser.add_bool_arg('--search', default=True, help='search down in density rather than assume bracket')
parser.add_bool_arg('--condor', short_opt='-j', default=False, help='create a condor job')
parser.add_bool_arg('--dagman', short_opt='-d', default=True, help='create a DAGMan job to run the condor job')
parser.add_bool_arg('--clean', short_opt='-c', default=True, help='clean up intermediate files')
parser.add_bool_arg('--run', short_opt='-x', default=False, help='run the condor or DAGMan job')
parser.add_bool_arg('--notify', default=True, help='notify on completion of DAGMan job')
args = parser.parse_args()

args.script = os.path.basename(__file__)
args.executable = sys.executable

opts = [f'--n={args.n}', f'--T={args.T}', f'--rho={args.rho}',
        # f'--A={args.A}', f'--B={args.B}',
        f'--ptol={args.ptol}', f'--ns={args.ns}', f'--np={args.np}',
        '--cold' if args.cold else '--no-cold',
        '--relative' if args.relative else '--no-relative',
        '--search' if args.search else '--no-search']

T_vals = pyHNC.as_linspace(args.T)

if args.condor: # create scripts to run jobs then exit

    njobs = len(T_vals)

    condor_job = f'{args.job_name}.job'
    lines = ['should_transfer_files = YES',
             'when_to_transfer_output = ON_EXIT',
             'notification = never',
             'universe = vanilla',
             f'opts = ' + ' '.join(opts),
             f'transfer_input_files = pyHNC.py,{args.script}',
             f'executable = {args.executable}',
             f'arguments = {args.script} {args.job_name} $(opts) --process=$(Process)',
             f'output = {args.job_name}__$(Process).out',
             f'error = {args.job_name}__$(Process).err',
             f'queue {njobs}']
    with open(condor_job, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    scripts = [condor_job]

    if args.dagman:
        dag_job = f'{args.job_name}__dag.job'
        post_script = f'{args.job_name}__script.sh'
        with open(dag_job, 'w') as f:
            f.write(f'JOB A {condor_job}\n')
            f.write(f'SCRIPT POST A /usr/bin/bash {post_script}\n')
        with open(post_script, 'w') as f:
            for k in range(njobs):
                redirect = '> ' if k == 0 else '>>'
                f.write(f'cat {args.job_name}__{k:d}.dat {redirect} {args.job_name}.dat\n')
            if args.clean:
                for k in range(njobs):
                    for ext in ['out', 'err', 'dat']:
                        f.write(f'rm -f {args.job_name}__{k:d}.{ext}\n')
            if args.notify:
                f.write(f'notify-send -u low -i info "{args.job_name} DAGMan job finished"\n')
        scripts = scripts + [dag_job, post_script]
        run_command = f'condor_submit_dag -notification Never {dag_job}'
    else:
        run_command = f'condor_submit {condor_job}'

    print(f'{args.script}: created', ', '.join(scripts))

    if args.run: # run if required, else print out the required command
        subprocess.call(run_command, shell=True)
    else:
        print(run_command)

    exit(0)

# *** Main computation starts ***

try:

    # The following are Tables from Sokhan et al., Soft Matter 19, 5824 (2023).

    Table1 = {'2': (2, 25.0, 3.02), # n, A, B, values
              '3': (3, 15.0, 7.2),
              '4': (4, 10.0, 15.0)}

    Table2 = {'2': (1.025, 0.2951, 0.519), # T_c, p_c, ρ_c values
              '3': (1.283, 0.3979, 0.504),
              '4': (1.290, 0.4095, 0.484)}

    if args.n in Table1:
        n, A, B = Table1[args.n] # unpack the default values
        Tc, _, ρcσ3 = Table2[args.n] # where available
    else:
        allowed_n = ', '.join(Table1.keys())
        raise NotImplementedError(f'n restricted to {allowed_n}')

    # Fish out the right temperature in units of kB in map/reduce mode,
    # else use the first value.

    k = args.process if args.process is not None else 0
    T = T_vals[k] * multiply_by(Tc, iff=args.relative)
    β = 1 / T

    A = args.A if args.A is not None else A # overwrite if necessary
    B = args.B if args.B is not None else B # overwrite if necessary
    σ = 1 - ((n+1)/(2*B))**(1/(n-1)) # the size is where the potential vanishes

    ρc = ρcσ3 / σ**3 # back out the critical value

    # density, temperature

    ρ_vals = pyHNC.as_linspace(args.rho) * multiply_by(ρc, iff=args.relative)

    if len(ρ_vals) == 1:
        ρ = ρ_vals[0]
    else:
        ρ1, ρ2 = ρ_vals.tolist()[0:2]

    grid = pyHNC.Grid(**pyHNC.grid_args(args)) # make the initial working grid

    r, Δr = grid.r, grid.deltar # extract the co-ordinate array for use below

    if args.verbose:
        print(f'{args.script}: {grid.details}')

    # Define the nDPD potential as in Eq. (5) in Sokhan et al., assuming
    # r_c = 1, and the force (negative derivative), then solve the HNC
    # problem.  The arrays here are all size ng-1, same as r[:].

    φ = truncate_to_zero(A*B/(n+1)*(1-r)**(n+1) - A/2*(1-r)**2, r, 1)
    f = truncate_to_zero(A*B*(1-r)**n - A*(1-r), r, 1)

    solver = pyHNC.PicardHNC(grid, nmonitor=500, **pyHNC.solver_args(args))

    if args.verbose:
        print(f'{args.script}: {solver.details}')

    # The excess virial pressure p = ρ + 2πρ²/3 ∫_0^∞ dr r³ f(r) h(r),
    # where f(r) = −dφ/dr is the force: see Eq. (2.5.22) in Hansen &
    # McDonald, "Theory of Simple Liquids" (3rd edition).

    # The mean-field contribution is 2πρ²/3 ∫_0^∞ dr r³ f(r) = ∫_0^1 dr r³
    # [AB(1−r)^n − A(1−r)] = πAρ²/30 * [120B/((n+1)(n+2)(n+3)(n+4)) − 1].

    def pressure(ρ):
        for second_attempt in [False, True]:
            if second_attempt or args.cold: # try again from cold start
                solver.warmed_up = False
            soln = solver.solve(β*φ, ρ, monitor=args.verbose) # solve model at β = 1/T
            if soln.converged:
                break
        else:
            raise RecursionError('failed to converge after two attempts')
        h = soln.hr
        p_mf = π*A*ρ**2/30*(120*B/((n+1)*(n+2)*(n+3)*(n+4)) - 1)
        p_xc = 2/3*π*ρ**2 * np.trapz(r**3*f*h, dx=Δr)
        p_ex = p_mf + p_xc
        p = ρ*T + p_ex
        return p

    print(f'{args.script}:', ' '.join(opts))
    print(f'{args.script}: model: nDPD with n = {n:d}, A = {A:g}, B = {B:g}, σ = {σ:g}, T = {T/Tc:g}')

    if len(ρ_vals) == 1: # range finding exercise, not encountered in normal use
        p = pressure(ρ)
        print(f'{args.script}: ρ/ρc, p =\t{ρ/ρc:g}\t{p:g}')
        exit(0)

    p1 = pressure(ρ1)
    p2 = pressure(ρ2)

    if p1*p2 > 0.0:
    
        if not args.search:
            raise RecursionError('root not bracketed and no search requested')

        Δρ = ρ2 - ρ1 # search up and down in density with this step size

        for i in range(args.ns):
            if p1*p2 < 0:
                break
            if p1 > 0:
                ρ1, ρ2 = ρ1-Δρ, ρ1
                p1, p2 = pressure(ρ1), p1
            else:
                ρ1, ρ2 = ρ2, ρ2+Δρ
                p1, p2 = p2, pressure(ρ2)
            print(f'{args.script}: search {i:3d}, ρ/ρc, p =\t{ρ1/ρc:g}\t{ρ2/ρc:g}\t{p1:g}\t{p2:g}')
        else:
            raise RecursionError('root not bracketed and search exhausted')

    # Bracketed root, proceed to find where the pressure vanishes 

    print(f'{args.script}: iteration 000, ρ/ρc, p =\t\t{ρ1/ρc:g}\t\t{p1:g}')
    print(f'{args.script}: iteration  00, ρ/ρc, p =\t\t{ρ2/ρc:g}\t\t{p2:g}')

    for i in range(args.np):
        #ρ = 0.5*(ρ1 + ρ2) # interval halving
        ρ = (p1*ρ2 - p2*ρ1) / (p1 - p2) # secant method
        p = pressure(ρ)
        print(f'{args.script}: iteration {i:3d}, ρ/ρc, p =\t{ρ1/ρc:g}\t{ρ/ρc:g}\t{ρ2/ρc:g}\t{p:g}')
        ρ1, ρ2, p1, p2 = (ρ, ρ2, p, p2) if p*p1 > 0.0 else (ρ1, ρ, p1, p)

except (RuntimeError, RecursionError) as err:
    failure = str(err)
    p = None

success = 'success' if p is not None and abs(p) < args.ptol else 'improperly converged'

if failure:
    print(f'{args.script}: {failure}')
else:
    print(f'{args.script}: {success}: T, ρ/ρc, p =\t{T/Tc:g}\t{ρ/ρc:g}\t{p:g}')

    
if args.job_name:

    sub = '' if args.process is None else f'__{args.process:d}' # double underscore here
    data_file = f'{args.job_name}{sub}.dat' # only one data file
    with open(data_file, 'w') as f:
        data = {'n': n, 'A': A, 'B': B, 'T': T/Tc, 'rho': 0, 'pressure': 0}
        if args.process is None or args.process == 0: # write for first file
            f.write(f'# ./{args.script} ' + ' '.join(opts) + '\n')
            f.write('# ' + '\t'.join([f'{key}({i+1})' for i, key in enumerate(data.keys())]) + '\tfile\tstatus\n')
        if failure: # log an entry anyways
            f.write('# ' + '\t'.join([('%g' % data[key]) for key in data]) + f'\t\t{data_file}\t{failure})\n')
        else:
            data['rho'] = ρ/ρc
            data['pressure'] = p
            f.write('\t'.join([('%g' % data[key]) for key in data]) + f'\t{data_file}\t{success}\n')

    print(f'{args.script}: result saved to {data_file}')
