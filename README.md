# CartilageInverseFE

Python scripts for **inverse finite element parameter identification** of articular
cartilage from stress-relaxation micro-indentation. Given experimental
force–displacement data, the pipeline drives FEBio simulations in a
finite-element-in-the-loop optimization to identify the constitutive parameters
of viscoelastic (VE), biphasic poroelastic (BPE), and fiber-reinforced BPE models.

> ⚠️ Work in progress — the optimization algorithm is under active development.

## Overview
- Displacement-controlled loading built per specimen from experimental data
  (normalized loadcurve + peak amplitude), written once per specimen.
- Material parameters substituted per iteration; forward solve in FEBio 4.
- Objective: weighted ramp / hold / peak force mismatch.
- Optimizer: Nelder–Mead (scipy), warm-start chaining across specimens,
  5-start fallback for flagged specimens.
- Plot code for the optimization results
- Plot code for parameter space ( optional to include )
- parameter sweep code(optional to include)
## Requirements
- Python 3.x, numpy, pandas, scipy
- FEBio 4

## Status / TODO
- [ ] Finalize objective weighting and tolerances （change into even weight as suggested)
- [ ] Clean hardcoded paths → config
- [ ] Add usage instructions

## Author
Keke Xia 
