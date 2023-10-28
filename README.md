## Hyper-netted chain (HNC) solver for Ornstein-Zernike (OZ) equation

_Current version:_

v1.0 - initial working version

### Summary

Implements a python module `pyHNC` for solving the Ornstein-Zernike (OZ)
equation using the hypernetted-chain (HNC) closure, for
single-component systems, with soft potentials (no hard cores).  It
uses the [FFTW](https://www.fftw.org/) library to do the Fourier
transforms, accessed _via_ the [pyFFTW](https://pyfftw.readthedocs.io/en/latest/)
wrapper.

The code is intended to be partly pedagogical.  It illustrates how to
implement three-dimensional Fourier-Bessel transforms using FFTW (see
also below).

The code currently comprises:

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
Simplifications compared to
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC) include
the fact that hard cores are not implemented, only a single component
is assumed, and Picard iteration is used rather than the Ng
accelerator.

### HNC closure of the OZ equation

What's being solved here is the Ornstein-Zernike (OZ) equation in
reciprocal space in the form _h_(_q_) = _c_(_q_) + ρ _h_(_q_)
_c_(_q_), in combination with the HNC closure in real space as
_g_(_r_) = exp[ − _v_(_r_) + _h_(_r_) − _c_(_r_)], using Picard
iteration.

Here _c_(_r_) is the direct correlation function, _h_(_r_) = _g_(_r_)
− 1 is the total correlation function, and _v_(_r_) is the potential.
In practice the OZ equation and the HNC closure are written and solved
iteratively in terms of the indirect correlation function _e_(_r_) =
_h_(_r_) − _c_(_r_).  An initial guess if the solver is not warmed up
is _c_(_r_) = − _v_(_r_) (this is the random phase approximation or
RPA,and for systems without hard cores is equivalent to the
mean-spherical approximation or MSA).

### FFTW and Fourier-Bessel transforms

The Fourier-Bessel forward transform of a function _f_(_r_) is

_g_(_q_) = 4π/_q_ ∫<sub>0</sub><sup>∞</sup>
d<em>r</em> sin(_qr_) _r_ _f_(_r_) .

From the FFTW [documentation](https://www.fftw.org/fftw3_doc/1d-Real_002dodd-DFTs-_0028DSTs_0029.html), `RODFT00` implements

_Y_<sub>_k_</sub> = 2 ∑<sub>_j_=0</sub><sup>_n_−1</sup>
_X_<sub>_j_</sub> sin[π(_j_+1)(_k_+1)/(_n_+1)] ,

where _n_ is the common length of the arrays _X_<sub>_j_</sub> and
_Y_(_k_)_Y_<sub>_k_</sub>.  To cast this into the right form, set
Δ<em>r</em> × Δ<em>q</em> = π / (_n_+1) and assign _r_<sub>_j_</sub> = (_j_+1)
× Δ<em>r</em> for _j_ = 0 to _n_−1, and likewise _q_<sub>_k_</sub> = (_k_+1) ×
Δ<em>q</em> for _k_ = 0 to _n_−1, so that

_Y_<sub>_k_</sub> = 2 ∑<sub>_j_=0</sub><sup>_n_−1</sup>
_X_<sub>_j_</sub> sin(_r_<sub>_j_</sub> _q_<sub>_k_</sub>) .

In terms of the desired integral we finally have

_g_(_q_<sub>_k_</sub>) = 2 π Δ<em>r</em> / _q_<sub>_k_</sub>
× 2 ∑<sub>_j_=0</sub><sup>_n_−1</sup>
(_r_ _f_)<sub>_j_</sub>
sin(_r_<sub>_j_</sub> _q_<sub>_k_</sub>) .

It is this which is implemented in the code.
The Fourier-Bessel back transform

_f_(_r_) = 1/(2π²<em>r</em>) ∫<sub>0</sub><sup>∞</sup>
d<em>q</em> sin(_qr_) _q_ _g_(_q_)

is handled similarly.

### On FFTW efficiency

Timing test (below) indicate that FFTW is very fast when the array
length in the above is a power of two _minus one_, which doesn't quite
seem to fit with the
[documentation](https://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transforms.html).
Here, the grid size in pyHNC is typically a power of two, but the
arrays passed to FFTW are one less than this in length.  Some typical
timing results on a moderately fast Intel NUC11TZi7 (11th Gen Intel
Core i7-1165G7 @ 2.80GHz) support this:
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

This program is copyright &copy; 2023 Patrick B Warren.  
