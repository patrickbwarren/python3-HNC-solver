## Hyper-netted chain (HNC) solver for Ornstein-Zernike (OZ) equation

_Current version:_

v1.0 - initial working version

### Summary

Implements a python module `pyHNC` for solving the Ornstein-Zernike
(OZ) equation using the hypernetted-chain (HNC) closure, for
single-component systems, with soft potentials (no hard cores) such as
dissipative particle dynamics (DPD).  It uses the
[FFTW](https://www.fftw.org/) library to do the Fourier transforms,
accessed _via_ the [pyFFTW](https://pyfftw.readthedocs.io/en/latest/)
wrapper.

The code is intended for rapid prototyping, but is also partly
pedagogical with the intent of attempting to capture some of the
[tacit knowledge](https://en.wikipedia.org/wiki/Tacit_knowledge)
involved in undertaking this kind of calculation. It currently comprises:

* `pyHNC.py` : python module implementing the functionality;
* `dpd_demo.py` : demonstrate the capabilities for standard DPD;
* `dpd_gw_compare.py` : compare DPD equation of state to literature;
* `fftw_test.py` : test FFTW for Fourier-Bessel transforms.

For more details see extensive comments in the codes, and also the
documentation for the parallel
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC) project.
The book "Theory of Simple Liquids" by Jean-Pierre Hansen and Ian
R. McDonald is foundational -- either the
[3rd edition](https://shop.elsevier.com/books/theory-of-simple-liquids/hansen/978-0-12-370535-8) (2006)
or the [4th edition](https://www.sciencedirect.com/book/9780123870322/theory-of-simple-liquids) (2013).

Simplifications compared to the (faster)
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC) include
the fact that hard cores are not implemented, only a single component
is assumed, and simple Picard iteration is used rather than the Ng
accelerator.  The present code is also implemented entirely in python,
rather than
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC) which is
mostly implemented in FORTRAN 90.

The `--sunlight` option for `dpd_demo.py` compares the present
implementation to results from
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC).  For
this to work, the compiled python interface (`*.pyf`) and shared
object (`*.so`) dynamically-linked library from
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC) should be
made acessible to `dpd_demo.py`.  This can be done for example by
copying `oz.pyf` and `oz.*.so` from
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC)
(avaliable after running `make`) to the directory containing
`dpd_demo.py`.

### HNC closure of the OZ equation

What's being solved here is the Ornstein-Zernike (OZ) equation in
reciprocal space in the form <em>h</em>(<em>q</em>) =
<em>c</em>(<em>q</em>) + ρ <em>h</em>(<em>q</em>)
<em>c</em>(<em>q</em>), in combination with the hypernetted-chain
(HNC) closure in real space as <em>g</em>(<em>r</em>) = exp[ −
<em>v</em>(<em>r</em>) + <em>h</em>(<em>r</em>) −
<em>c</em>(<em>r</em>)], using Picard iteration.

Here ρ is the number density, <em>v</em>(<em>r</em>) is the potential
in units of <em>k</em><sub>B</sub><em>T</em>, <em>g</em>(<em>r</em>)
is the pair correlation function, <em>h</em>(<em>r</em>) =
<em>g</em>(<em>r</em>) − 1 is the total correlation function, and
<em>c</em>(<em>r</em>) is the direct correlation function which is
defined by the OZ equation.  In practice the OZ equation and the HNC
closure are written and solved iteratively (see next) in terms of the
indirect correlation function <em>e</em>(<em>r</em>) =
<em>h</em>(<em>r</em>) − <em>c</em>(<em>r</em>).

An initial guess if the solver is not warmed up is
<em>c</em>(<em>r</em>) = − <em>v</em>(<em>r</em>) ; this is the
random-phase approximation (RPA), which for systems without hard cores
is equivalent to the mean spherical approximation (MSA).

### Algorithm

Given an initial guess <em>c</em>(<em>r</em>), the solver implements the
following scheme (<em>cf</em> [SunlightHNC](https://github.com/patrickbwarren/SunlightHNC)):

* Fourier-Bessel forward transform <em>c</em>(<em>r</em>) →
  <em>c</em>(<em>q</em>) ;
* solve the OZ equation for <em>e</em>(<em>q</em>) =
  <em>c</em>(<em>q</em>) / [1 − ρ <em>c</em>(<em>q</em>)] −
  <em>c</em>(<em>q</em>) ;
* Fourier-Bessel back transform <em>e</em>(<em>q</em>) →
  <em>e</em>(<em>r</em>) ;
* implement the HNC closure as <em>c</em>'(<em>r</em>) =
  exp[ − <em>v</em>(<em>r</em>) + <em>e</em>(<em>r</em>)] −
  <em>e</em>(<em>r</em>) − 1 ;
* replace <em>c</em>(<em>r</em>) by
  α <em>c</em>′(<em>r</em>) + (1−α) <em>c</em>(<em>r</em>) (Picard
  mixing step);
* check for convergence by comparing <em>c</em>(<em>r</em>) and
  <em>c'</em>(<em>r</em>) ;
* if not converged, repeat.

Typically this works for a Picard mixing fraction α = 0.2, and for
standard DPD for example convergence to an accuracy of
10<sup>−12</sup> for a grid size <em>N</em><sub>g</sub> =
2<sup>12</sup> = 4096 (see below!) with a grid spacing Δ<em>r</em> = 0.01 is
achieved with a few hundred iterations (a fraction of a second CPU
time).

Once converged, the pair correlation function and static structure
factor can be found from:

* <em>g</em>(<em>r</em>) = 1 + <em>h</em>(<em>r</em>) where
  <em>h</em>(<em>r</em>) = <em>e</em>(<em>r</em>) +
  <em>c</em>(<em>r</em>) ;
* <em>S</em>(<em>q</em>) = 1 + ρ <em>h</em>(<em>q</em>) where
  <em>h</em>(<em>q</em>) = <em>e</em>(<em>q</em>) +
  <em>c</em>(<em>q</em>) .

Thermodynamic quantities can also now be computed, for example the
energy density and virial pressure follow from Eqs. (2.5.20) and (2.5.22) in
Hansen and McDonald, "Theory of Simple Liquids" (3rd edition) as:

* <em>e</em> = 2πρ² ∫<sub>0</sub><sup>∞</sup> d<em>r</em> <em>r</em>²
  <em>v</em>(<em>r</em>) <em>g</em>(<em>r</em>) ;
* <em>p</em> = ρ + 2πρ²/3 ∫<sub>0</sub><sup>∞</sup> d<em>r</em> <em>r</em>³
  <em>f</em>(<em>r</em>) <em>g</em>(<em>r</em>) where
  <em>f</em>(<em>r</em>) = − d<em>v</em>/d<em>r</em> .

In practice these should usually be calculated with
<em>h</em>(<em>r</em>) = <em>g</em>(<em>r</em>) − 1, since the
mean-field estimates (<em>i. e.</em> the above with
<em>g</em>(<em>r</em>) = 1) can usually be calculated analytically.
Note that in this case an integration by parts shows that the two
integrals are actually the same, and are essentially equal to the
value of the potential at the origin in reciprocal space: 2πρ²/3
∫<sub>0</sub><sup>∞</sup> d<em>r</em> <em>r</em>³
<em>f</em>(<em>r</em>) = 2πρ² ∫<sub>0</sub><sup>∞</sup> d<em>r</em>
<em>r</em>² <em>v</em>(<em>r</em>) = ρ²/2 ∫ d³<b>r</b>
<em>v</em>(<em>r</em>) = ρ²/2 <em>v</em>(<em>q</em>=0)

### FFTW and Fourier-Bessel transforms

The code illustrates how to implement three-dimensional
Fourier-Bessel transforms using FFTW.
The Fourier-Bessel forward transform of a function
<em>f</em>(<em>r</em>) in three dimensions is (see [SunlightHNC](https://github.com/patrickbwarren/SunlightHNC) documentation):

<em>g</em>(<em>q</em>) = 4π / <em>q</em> ∫<sub>0</sub><sup>∞</sup>
d<em>r</em> sin(<em>qr</em>) <em>r</em> <em>f</em>(<em>r</em>) .

From the [FFTW documentation](https://www.fftw.org/fftw3_doc/1d-Real_002dodd-DFTs-_0028DSTs_0029.html),
`RODFT00` implements

<em>Y</em><sub><em>k</em></sub> = 2
∑<sub><em>j</em>=0</sub><sup><em>n</em>−1</sup>
<em>X</em><sub><em>j</em></sub> sin[π (<em>j</em>+1) (<em>k</em>+1) /
(<em>n</em>+1)] ,

where <em>n</em> is the common length of the arrays
<em>X</em><sub><em>j</em></sub> and <em>Y</em><sub><em>k</em></sub>.
To cast this into the right form, set Δ<em>r</em> × Δ<em>q</em> = π /
(<em>n</em>+1) and assign <em>r</em><sub><em>j</em></sub> =
(<em>j</em>+1) × Δ<em>r</em> for <em>j</em> = 0 to <em>n</em>−1, and
likewise <em>q</em><sub><em>k</em></sub> = (<em>k</em>+1) ×
Δ<em>q</em> for <em>k</em> = 0 to <em>n</em>−1, so that

<em>Y</em><sub><em>k</em></sub> = 2
∑<sub><em>j</em>=0</sub><sup><em>n</em>−1</sup>
<em>X</em><sub><em>j</em></sub> sin(<em>r</em><sub><em>j</em></sub>
<em>q</em><sub><em>k</em></sub>) .

For the desired integral we can then write

<em>g</em>(<em>q</em><sub><em>k</em></sub>) = 2 π Δ<em>r</em> /
<em>q</em><sub><em>k</em></sub> × 2
∑<sub><em>j</em>=0</sub><sup><em>n</em>−1</sup>
<em>r</em><sub><em>j</em></sub>
<em>f</em>(<em>r</em><sub><em>j</em></sub>)
sin(<em>r</em><sub><em>j</em></sub> <em>q</em><sub><em>k</em></sub>) ,

with the factor after the multiplication sign being calculated by
`RODFT00`.

The Fourier-Bessel back transform

<em>f</em>(<em>r</em>) = 1 / (2π²<em>r</em>) ∫<sub>0</sub><sup>∞</sup>
d<em>q</em> sin(<em>qr</em>) <em>q</em> <em>g</em>(<em>q</em>)

is handled similarly.

### On FFTW efficiency

Timing tests (below) indicate that FFTW is very fast when the array
length in the above is a power of two _minus one_, which doesn't quite
seem to fit with the
[documentation](https://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transforms.html).
Here, the grid size <em>N</em><sub>g</sub> in pyHNC is typically a power of two, but the
arrays passed to FFTW are shortened to <em>N</em><sub>g</sub> − 1.  Some typical
timing results on a moderately fast [Intel<sup>®</sup> NUC11TZi7](https://www.intel.com/content/www/us/en/products/sku/205605/intel-nuc-11-pro-kit-nuc11tnhi7/specifications.html) with an [11th Gen Intel<sup>®</sup>
Core™ i7-1165G7](https://www.intel.com/content/www/us/en/products/sku/205605/intel-nuc-11-pro-kit-nuc11tnhi7/specifications.html) processor (up to 4.70GHz) support this.  For example
```
$ time ./fftw_test.py --ng=4096 --deltar=0.01 --iter=500
ng, Δr, Δq, iters = 4096 0.01 0.07669903939428206 500
FFTW array sizes = 4095
real	0m0.336s
user	0m0.427s
sys	0m0.575s

$ time ./fftw_test.py --ng=4095 --deltar=0.01 --iter=500
ng, Δr, Δq, iters = 4095 0.01 0.07671776931843206 500
FFTW array sizes = 4094
real	0m0.376s
user	0m0.491s
sys	0m0.556s

$ time ./fftw_test.py --ng=4097 --deltar=0.01 --iter=500
ng, Δr, Δq, iters = 4097 0.01 0.07668031861337059 500
FFTW array sizes = 4096
real	0m0.521s
user	0m0.668s
sys	0m0.521s
```
With a grid of order one million grid points,
```
$ time ./fftw_test.py --ng=2^20 --deltar=1e-3
ng, Δr, Δq, iters = 1048576 0.001 0.0029960562263391427 10
FFTW array sizes = 1048575
real   0m0.946s
user   0m1.065s
sys    0m0.539s

$ time ./fftw_test.py --ng=2^20-1 --deltar=1e-3
ng, Δr, Δq, iters = 1048575 0.001 0.0029960590836037413 10
FFTW array sizes = 1048574
real   0m1.807s
user   0m1.876s
sys    0m0.565s

$ time ./fftw_test.py --ng=2^20+1 --deltar=1e-3
ng, Δr, Δq, iters = 1048577 0.001 0.0029960533690799943 10
FFTW array sizes = 1048576
real   0m3.106s
user   0m3.063s
sys    0m0.706s
```
In the code, FFTW is set up with the most basic `FFTW_ESTIMATE`
[planner flag](https://www.fftw.org/fftw3_doc/Planner-Flags.html).
This may make a difference in the end, but timing tests indicate that
with a power of two as used here, it takes much longer for FFTW to
find an optimized plan, than it does if it just uses the simple
heuristic implied by `FFTW_ESTIMATE`.  Obviously some further
investigations could be undertaken into this aspect.

The TL;DR take-home message here is _use a power of two_ for the
<em>N</em><sub>g</sub> parameter in the code!

### Copying

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see
<http://www.gnu.org/licenses/>.

### Copyright

This program is copyright &copy; 2023 Patrick B Warren (STFC).  

### Contact

Send email to patrick.warren{at}stfc.ac.uk.
