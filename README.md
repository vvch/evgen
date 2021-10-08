Event Generator for CLAS Analysis Framework
===========================================

Installation
------------

To run these programs you need

* `python` programming language version 3.8+
* `pip` package manager for python
* `ROOT` library (by CERN) with `python` bindings (necessary for some parts only, like histograms and LUND file producing; event generator itself can work without ROOT)
* to install python dependencies, run  
    `pip3 -r requirements.txt`



Event generator consists of several programs.


evgen.py
--------

The main program, generates events using differential cross-section
calculated from helicity amplitudes for 4 channels

* π⁰ p
* π⁺ n
* π⁰ n
* π⁻ p

in the kinematics range

* 0 ≤ Q^2 ≤ 5 GeV^2
* 1.08 ≤ W ≤ 2 GeV  (by MAID)
* 2 GeV < W  (by Piter Kroll, currently unavailable)

To get information about using command line parameters and configuration file options, run

```
./evgen.py --help
```

Configuration file `evgen.conf` may contain same options which are allowed in the command line, in the format `option=value`. Options specified in the command line override those in the config file.


hist.py
-------

Can be used to plot histograms from text files with columns: W, Q^2, cos θ, φ
To get information on using command line parameters, run

```
./hist.py --help
```

lund.py
-------

Can be used to convert text files with W, Q^2, cos θ, φ  columns to LUND format.
To get information on using command line parameters, run

```
./lund.py --help
```

sig_interpolate.py
------------------

Produces 3D plot of interpolated differential cross-section in specified W, Q^2 point.

To get information on using command line parameters, run

```
./sig_interpolate.py --help
```
