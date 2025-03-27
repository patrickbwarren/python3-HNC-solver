## Hyper-netted chain (HNC) solver for Ornstein-Zernike (OZ) equation

_Current version:_ v1.0 - initial working version

### Summary

Implements a python module `pyHNC` for solving the Ornstein-Zernike
(OZ) equation using the hypernetted-chain (HNC) closure, for
single-component systems, with soft potentials (no hard cores) such as
dissipative particle dynamics (DPD).  It uses the
[FFTW](https://www.fftw.org/) library to do the Fourier transforms,
accessed _via_ the [pyFFTW](https://pyfftw.readthedocs.io/en/latest/)
wrapper, together with basic [NumPy](https://numpy.org/) function
calls.

The code is intended for rapid prototyping, but is also partly
pedagogical with the intent of attempting to capture some of the
[tacit knowledge](https://en.wikipedia.org/wiki/Tacit_knowledge)
involved in undertaking this kind of calculation.

Basic codes in the repository include:

* `pyHNC.py` : python module implementing the functionality;
* `fftw_demo.py` : test FFTW for Fourier-Bessel transforms;
* `dpd_demo.py` : demonstrate the capabilities for standard DPD;
* `dpd_eos.py` : calculate data for standard DPD equation of state (EoS);
* `dpd_gw_compare.py` : compare to [Groot and Warren, J. Chem. Phys.
   **107**, 4423 (1997)](https://doi.org/10.1063/1.474784).

For more details see extensive comments in the codes, and also the
documentation for the parallel
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC) project.
The book "Theory of Simple Liquids" by Jean-Pierre Hansen and Ian
R. McDonald is foundational -- either the [3rd
edition](https://shop.elsevier.com/books/theory-of-simple-liquids/hansen/978-0-12-370535-8)
(2006) or the [4th
edition](https://www.sciencedirect.com/book/9780123870322/theory-of-simple-liquids)
(2013).

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

The other codes are all experimental, and under development:

* `ndpd_demo.py` : *n*DPD, as in [Sokhan *et al.*,
   Soft Matter **19**, 5824 (2023)](https://doi.org/10.1039/D3SM00835E);
* `ndpd_rpa.py` : implement the RPA and EXP approximations for the EoS;
* `ndpd_liquidus.py` : estimate the liquidus as the point where *p* = 0; condor-enabled;
* `mdpd_hnc.py` : various HNC variants for many-body (MB) DPD; condor-enabled;
* `mdpd_dft.py` : 'vanilla' DFT for MB DPD;
* `mdpd_percus.py` : Percus-like DFT for MB DPD;
* `timing.py` : for measuring performance of condor DAGMan jobs.

Other files herein:

* `solute_demo.ipynb` : jupyter notebook illustrating solute calculations;
* `mu_dpd_hendrikse2025.csv` : solute excess chemical potential data digitised from Fig. 1 of
[Hendrikse *et al.*, Phys. Chem. Chem. Phys. **27**, 1554-66 (2025)](https://doi.org/10.1039/D4CP03791J).

### What's being solved here?

#### HNC closure of the OZ equation

What's being solved here is the Ornstein-Zernike (OZ) equation in
reciprocal space in the form

* *h*(*q*) = *c*(*q*) + ρ *h*(*q*) *c*(*q*) ,

in combination with the hypernetted-chain (HNC) closure in real space
as

* *g*(*r*) = exp[ − *v*(*r*) + *h*(*r*) − *c*(*r*)] ,

using Picard iteration.

Here ρ is the number density, *v*(*r*) is the potential in units of
*k*<sub>B</sub>*T*, *g*(*r*) is the pair correlation function,
*h*(*r*) = *g*(*r*) − 1 is the total correlation function, and
*c*(*r*) is the direct correlation function which is defined by the OZ
equation.  In practice the OZ equation and the HNC closure are written
and solved iteratively (see next) in terms of the indirect correlation
function *e*(*r*) = *h*(*r*) − *c*(*r*).

An initial guess if the solver is not warmed up is *c*(*r*) = −
*v*(*r*) ; this is the random-phase approximation (RPA), which for
systems without hard cores is equivalent to the mean spherical
approximation (MSA).

#### Algorithm

Given an initial guess *c*(*r*), the solver implements the following
scheme (*cf*
[SunlightHNC](https://github.com/patrickbwarren/SunlightHNC)):

* Fourier-Bessel forward transform *c*(*r*) → *c*(*q*) ;
* solve the OZ equation for *e*(*q*) = *c*(*q*) / [1 − ρ *c*(*q*)] −
  *c*(*q*) ;
* Fourier-Bessel back transform *e*(*q*) → *e*(*r*) ;
* implement the HNC closure as *c*'(*r*) = exp[ − *v*(*r*) + *e*(*r*)]
  − *e*(*r*) − 1 ;
* replace *c*(*r*) by α *c*'(*r*) + (1−α) *c*(*r*) (Picard mixing step);
* check for convergence by comparing *c*(*r*) and *c*'(*r*) ;
* if not converged, repeat.

Typically this works for a Picard mixing fraction α = 0.2, and for
standard DPD for example convergence to an accuracy of
10<sup>−12</sup> for a grid size *N*<sub>g</sub> = 2<sup>12</sup> =
8192 with a grid spacing Δ*r* = 0.02 is achieved with a few hundred
iterations (a fraction of a second CPU time).

### Structural properties

Once converged, the pair correlation function and static structure
factor can be found from:

* *g*(*r*) = 1 + *h*(*r*) where *h*(*r*) = *e*(*r*) + *c*(*r*) ;
* *S*(*q*) = 1 + ρ *h*(*q*) where *h*(*q*) = *e*(*q*) + *c*(*q*) .

### Thermodynamics

#### Energy density and virial pressure

Thermodynamic quantities can also now be computed, for example the excess
energy density and virial pressure follow from Eqs. (2.5.20) and (2.5.22) in
Hansen and McDonald, "Theory of Simple Liquids" (3rd edition) as:

* *e* = 3ρ/2 + 2πρ<sup>2</sup> ∫<sub>0</sub><sup>∞</sup> d*r*
  *r*<sup>2</sup> *v*(*r*) *g*(*r*) ;
* *p* = ρ + 2πρ<sup>2</sup>/3 ∫<sub>0</sub><sup>∞</sup> d*r*
  *r*<sup>3</sup> *f*(*r*) *g*(*r*) where *f*(*r*) = − d*v*/d*r* ,

where the first terms here are the ideal contributions, in units of
*k*<sub>B</sub>*T*.

In practice these should usually be calculated with *h*(*r*) =
*g*(*r*) − 1, since the mean-field contributions (*i. e.* the above
with *g*(*r*) = 1) can usually be calculated analytically.  Note that
in this case an integration by parts shows that the two integrals are
actually the same, and are essentially equal to the value of the
potential at the origin in reciprocal space: 2πρ<sup>2</sup>/3
∫<sub>0</sub><sup>∞</sup> d*r* *r*<sup>3</sup> *f*(*r*) =
2πρ<sup>2</sup> ∫<sub>0</sub><sup>∞</sup> d*r* *r*<sup>2</sup>
*v*(*r*) = ρ<sup>2</sup>/2 ∫ d<sup>3</sup>**r** *v*(*r*) =
ρ<sup>2</sup>/2 *v*(*q*=0).

#### Compressibility

Eq. (2.6.12) in Hansen and McDonald shows that in units of
*k*<sub>B</sub>*T* the isothermal compressibility satisfies ρ
χ<sub>T</sub> = 1 + 4πρ ∫<sub>0</sub><sup>∞</sup> d*r* *r*<sup>2</sup>
*h*(*r*) where χ<sub>T</sub> = − (1/*V*) ∂*V*/∂*p*.  In terms of the
EoS, this last expression can be written as χ<sub>T</sub><sup>−1</sup>
= ρ d*p*/dρ.  Further, in reciprocal space the OZ equation (above) can
be written as

* [1 + ρ *h*(*q*)] [1 − ρ *c*(*q*)] = 1 .

Employing this at *q* = 0, one therefore obtains

* d*p*/dρ = [ρχ<sub>T</sub>]<sup>−1</sup> = 1 − 4πρ
  ∫<sub>0</sub><sup>∞</sup> d*r* *r*<sup>2</sup> *c*(*r*) .

Given *c*(*r*) as a function of density, this can be integrated to
find *p*(ρ).  This is known as the compressibility route to the EoS.

#### Chemical potential

A result peculiar to the HNC is the closed-form expression for the
chemical potential given in Eq. (4.3.21) in Hansen and McDonald (I
thank Andrew Masters for drawing my attention to this),

* μ / *k*<sub>B</sub>*T* = ln ρ + 4πρ ∫<sub>0</sub><sup>∞</sup> d*r*
  *r*<sup>2</sup> [ ½ *h* (*h* − *c*) − *c* ] .

Here the reference standard state corresponds to ρ = 1.  Since the
Gibbs-Duhem relation in the form d*p* = ρ dμ can be integrated to find
the pressure, this affords another route to the EoS: the chemical
potential route.  In HNC the chemical potential should verify *f* −
ρμ + *p* = 0 where *f* is the free energy density.  This follows from
the fact that the grand potential Ω = *F* − μ*N* = −*pV*.  In the form
*f* = ρ μ − *p*, where *p* is the virial pressure, it can be used as a
direct method to access the free energy, in contrast to the more
generic the coupling constant integration method described next.

#### Free energy and coupling constant integration

It follows from the basic definition of the free energy *F* = − ln ∫
d<sup>*N*</sup>{**r**} e<sup>−*U*</sup> that ∂*F*/∂λ = ⟨∂*U*/∂λ⟩ where
λ is a parameter in the potential function *U*.

We can therefore calculate the free energy from *F* =
*F*<sub>0</sub> + ∫<sub>0</sub><sup>1</sup> dλ ⟨∂*U*/∂λ⟩<sub>λ</sub>.
If λ is simply a multiplicative scaling, *U* → λ *U*, then ∂*U*/∂λ =
*U* and we have a _coupling constant integration_ scheme *F* =
*F*<sub>0</sub> + ∫<sub>0</sub><sup>1</sup> dλ ⟨*U*⟩<sub>λ</sub> where
the indicated average should be taken with the potential energy scaled
by a factor λ. In this scheme *F*<sub>0</sub> is just the free energy
of an ideal gas of non-interacting particles since λ → 0 switches off
the interactions.

Since the free energy can be differentiated to find the pressure, this
is the basis for the so-called energy route to the EoS.  For example,
if the free energy density is available as a function of density,
*f*(ρ), the pressure follows from p = −∂*F*/∂*V* as *p* =
ρ<sup>2</sup> d(*f*/ρ)/dρ where *f*/ρ is the free energy per particle.
It also follows that the compressibility [ρχ<sub>T</sub>]<sup>−1</sup>
= ρ d<sup>2</sup>*f*/dρ<sup>2</sup>.

The mean-field contribution to this can be calculated immediately
since the contribution to the energy density 2πρ<sup>2</sup>
∫<sub>0</sub><sup>∞</sup> d*r* *r*<sup>2</sup> *v*(*r*) is independent
of λ and therefore ∫<sub>0</sub><sup>1</sup> dλ applied to this term
trivially evaluates to the same.  Furthermore, since this term is ∝
ρ<sup>2</sup>, following the indicated route to the pressure shows
that this exact same term appears there too.  So the mean-field
contribution to the pressure here is the same as the virial route
mean-field pressure.

For the non-mean-field correlation contribution we sketch the algorithm:

* solve the HNC closure of OZ equation for the _scaled_ pair potential
  λ*v*(*r*) to get *h*(*r*; λ) ;

* calculate ∆*e*(λ) = 2πρ<sup>2</sup> ∫<sub>0</sub><sup>∞</sup> d*r*
 *r*<sup>2</sup> *v*(*r*) *h*(*r*; λ) with the _unscaled_ pair
 potential;

* the excess correlation free energy is then the integral ∆*f* =
  ∫<sub>0</sub><sup>1</sup> dλ ∆*e*(λ) .

* the excess correlation pressure then follows from ∆p = ρ<sup>2</sup>
d(∆*f* / ρ)/dρ .  This should be added to the mean-field contribution
to obtain the excess pressure, and the whole added to the ideal
contribution to find the total pressure.

In practice the coupling constant integration can be performed by any
number of numerical quadrature methods but typically (for me!) a basic
trapezium rule suffices.  The derivative with respect to density would
usually be computed numerically too.

For the HNC closure, which is free energy based, in fact it should be
_exactly_ true that the energy route pressure is the same as the
virial route pressure, not just the mean-field contributions.  So
differences here are a test of the numerics rather than the physical
approximations.

### Solutes

The above methodology can be repurposed to solve also the case of an
infinitely dilute solute inside a solvent.  To do this we start from
the OZ equations for a two-component mixture and specialise to the
case where the density of the second component vanishes.  In this
limit the OZ equations partially decouple in the sense that the
solvent case reduces to the above one-component problem, which can be
solved as already indicated.  The off-diagonal OZ relations become

* *h*<sub>01</sub>(*q*) = *c*<sub>01</sub>(*q*) +
ρ<sub>0</sub> *h*<sub>01</sub>(*q*) *c*<sub>00</sub>(*q*) ,

* *h*<sub>01</sub>(*q*) = *c*<sub>01</sub>(*q*) +
ρ<sub>0</sub> *h*<sub>00</sub>(*q*) *c*<sub>01</sub>(*q*) .

The equivalence between the two can be proven from the OZ relation for
the solvent.  These should be supplemented by the HNC closure for the
off-diagonal case

* *g*<sub>01</sub>(*r*) = exp[ − *v*<sub>01</sub>(*r*) +
*h*<sub>01</sub>(*r*) − *c*<sub>01</sub>(*r*)] .

The second off-diagonal OZ relation can be written as 

* *h*<sub>01</sub>(*q*) = *c*<sub>01</sub>(*q*) 
(1 + ρ<sub>0</sub> *h*<sub>00</sub>(*q*)) 
= *S*<sub>00</sub>(*q*) *c*<sub>01</sub>(*q*) ,

where *S*<sub>00</sub>(*q*) is the solvent structure factor.  It
follows that the solute problem can be solved be re-purposing the
exact same algorithm as for the one-component system, replacing the OZ
equation step by the assignment,

* set *e*<sub>01</sub>(*q*) = *S*<sub>00</sub>(*q*) *c*<sub>01</sub>(*q*)
− *c*<sub>01</sub>(*q*) .

Applications of this infinitely-dilute solute limit are in the process
of being investigated.

### FFTW and Fourier-Bessel transforms

The code illustrates how to implement three-dimensional Fourier
transforms using FFTW.  The starting point is the Fourier transform
pair

*g*(**q**) = ∫ d<sup>3</sup>**r** 
*e*<sup>−*i***q**∙**r**</sup> *f*(**r**) ,

*f*(**r**) = 1 / (2π)<sup>3</sup> ∫ d<sup>3</sup>**q** 
*e*<sup>*i***q**∙**r**</sup> *g*(**q**) .

If the functions have radial symmetry, these reduce to the forward and
backward Fourier-Bessel transforms

*g*(*q*) = 4π / *q* ∫<sub>0</sub><sup>∞</sup> 
d*r* *r* *f*(*r*) sin *qr* ,

*f*(*r*) = 1 / (2π<sup>2</sup>*r*) ∫<sub>0</sub><sup>∞</sup> 
d*q* *q* *g*(*q*) sin *qr* .

From the [FFTW
documentation](https://www.fftw.org/fftw3_doc/1d-Real_002dodd-DFTs-_0028DSTs_0029.html),
`RODFT00` implements

*Y*<sub>*k*</sub> = 2 ∑<sub>*j*=0</sub><sup>*n*−1</sup>
*X*<sub>*j*</sub> sin[π (*j*+1) (*k*+1) / (*n*+1)] ,

where *n* is the common length of the arrays *X*<sub>*j*</sub> and
*Y*<sub>*k*</sub>.

To cast this into the right form, set Δ*r* × Δ*q* = π / (*n*+1) and
assign *r*<sub>*j*</sub> = (*j*+1) × Δ*r* for *j* = 0 to *n*−1, and
likewise *q*<sub>*k*</sub> = (*k*+1) × Δ*q* for *k* = 0 to *n*−1, so
that

*Y*<sub>*k*</sub> = 2 ∑<sub>*j*=0</sub><sup>*n*−1</sup>
*X*<sub>*j*</sub> sin *q*<sub>*k*</sub>*r*<sub>*j*</sub> .

For the desired Fourier-Bessel forward transform  we can then write

*g*(*q*<sub>*k*</sub>) = 2 π Δ*r* / *q*<sub>*k*</sub> × 2
∑<sub>*j*=0</sub><sup>*n*−1</sup> *r*<sub>*j*</sub>
*f*(*r*<sub>*j*</sub>) sin *q*<sub>*k*</sub>*r*<sub>*j*</sub> ,

with the factor after the multiplication sign being calculated by
`RODFT00`.

The Fourier-Bessel back transform is handled similarly.

#### On FFTW efficiency

Timing tests (below) indicate that FFTW is very fast when the array
length *n* in the above is a power of two _minus one_, which
doesn't quite seem to fit with the
[documentation](https://www.fftw.org/fftw3_doc/Real_002dto_002dReal-Transforms.html).
Hence, the grid size *N*<sub>g</sub> = *n* + 1 in pyHNC
is typically a power of two, but the arrays passed to FFTW are
shortened to *N*<sub>g</sub> − 1.  Some typical timing results
on a moderately fast [Intel<sup>®</sup>
NUC11TZi7](https://www.intel.com/content/www/us/en/products/sku/205605/intel-nuc-11-pro-kit-nuc11tnhi7/specifications.html)
with an [11th Gen Intel<sup>®</sup> Core™
i7-1165G7](https://www.intel.com/content/www/us/en/products/sku/205605/intel-nuc-11-pro-kit-nuc11tnhi7/specifications.html)
processor (up to 4.70GHz) support this.  For example
```
$ time ./fftw_demo.py --ng=8192 --deltar=0.02
ng, Δr, Δq, iters = 8192 0.02 0.019174759848570515 10
FFTW array sizes = 8191
real	0m0.321s
user	0m0.399s
sys	0m0.580s

$ time ./fftw_demo.py --ng=8193 --deltar=0.02
ng, Δr, Δq, iters = 8193 0.02 0.019172419465335 10
FFTW array sizes = 8192
real	0m0.347s
user	0m0.498s
sys	0m0.518s

$ time ./fftw_demo.py --ng=8191 --deltar=0.02
ng, Δr, Δq, iters = 8191 0.02 0.019177100803258414 10
FFTW array sizes = 8190
real	0m0.337s
user	0m0.457s
sys	0m0.547s
```
The same, but with 4.2 million grid points
```
$ time ./fftw_demo.py --ng=2^22 --deltar=1e-3
ng, Δr, Δq, iters = 4194304 0.001 0.0007490140565847857 10
FFTW array sizes = 4194303
real	0m4.087s
user	0m3.928s
sys	0m0.822s

$ time ./fftw_demo.py --ng=2^22+1 --deltar=1e-3
ng, Δr, Δq, iters = 4194305 0.001 0.0007490138780059611 10
FFTW array sizes = 4194304
real	0m10.682s
user	0m9.840s
sys	0m1.505s

$ time ./fftw_demo.py --ng=2^22-1 --deltar=1e-3
ng, Δr, Δq, iters = 4194303 0.001 0.0007490142351636954 10
FFTW array sizes = 4194302
real	0m14.539s
user	0m14.079s
sys	0m1.121s
```

In the code, FFTW is set up with the most basic `FFTW_ESTIMATE`
[planner flag](https://www.fftw.org/fftw3_doc/Planner-Flags.html).
This may make a difference in the end, but timing tests indicate that
with a power of two as used here, it takes much longer for FFTW to
find an optimized plan, than it does if it just uses the simple
heuristic implied by `FFTW_ESTIMATE`.  Obviously some further
investigations could be undertaken into this aspect.

The TL;DR take-home message here is _use a power of two_ for the
*N*<sub>g</sub> parameter in the code!

#### Choice of grid size

From above Δ*r* × Δ*q* = π / *N*<sub>g</sub> can be
inverted to suggest *N*<sub>g</sub> = π /
Δ*r* Δ*q*.  Since presumably we want the grid resolution
in real space and reciprocal space to be comparable, Δ*r* ≈
Δ*q*, and we want *N*<sub>g</sub> =
2<sup>*r*</sup>, this suggests the following table (where
Δ*q* is computed from Δ*r* and *N*<sub>g</sub>):
```
--deltar=0.05 --ng=2^11 (ng=2048 ⇒ Δq ≈ 0.031  )
--deltar=0.02 --ng=2^13 (ng=8192 ⇒ Δq ≈ 0.019  )
--deltar=0.01 --ng=2^15 (ng=32768 ⇒ Δq ≈ 0.0096 )
--deltar=5e-3 --ng=2^17 (ng=131072 ⇒ Δq ≈ 4.79e-3)
--deltar=2e-3 --ng=2^20 (ng=1048576 ⇒ Δq ≈ 1.50e-3)
--deltar=1e-3 --ng=2^22 (ng=4194304 ⇒ Δq ≈ 0.749e-3)
```

### Copying

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see
<http://www.gnu.org/licenses/>.

### Copyright

This program is copyright &copy; 2023 Patrick B Warren (STFC).  
Additional modifications copyright &copy; 2025 Joshua F Robinson (STFC).  

### Contact

Send email to patrick.warren{at}stfc.ac.uk.
