# CartilageInverseFE

Reference code for the **inverse finite-element (FE) parameter identification** part of
my MSc thesis on articular cartilage mechanics (University of Bern).

Constitutive parameters are identified from stress-relaxation micro-indentation data by
running FEBio forward simulations inside a Nelder–Mead optimization loop
(FE-in-the-loop). Two constitutive models are covered:

| Model | Folder | Identified parameters | Fixed parameters |
|---|---|---|---|
| **VE** – viscoelastic, neo-Hookean + 1-term Prony series | `reference_scripts/VE/` | `E`, `g1`, `t1` | `ν = 0.25` |
| **BPE + IsoFiber** – biphasic, neo-Hookean matrix + isotropic (spherical) continuous fibre distribution, constant permeability | `reference_scripts/BPE_IsoFiber/` | `E`, `ksi`, `k` (optimized as log10) | `ν = 0.25`, `φ0 = 0.2`, fibre `α = 0`, `β = 2` |

> **Note** – This repository documents *how* the optimization was done. It is not a
> packaged tool: the scripts are kept as they were run on the HPC cluster
> (absolute paths, SLURM settings, Chinese comments). To reuse them, edit the
> configuration block at the top of each script.

---

## Repository structure

```
CartilageInverseFE/
├── README.md
├── requirements.txt
├── docs/
│   └── optimization_method.md        # detailed description of the algorithm
├── reference_scripts/
│   ├── VE/
│   │   ├── Native_VE_singleStart_test.py     # step 1: single-start optimization, all samples
│   │   ├── Analyze_suspicious.py             # step 2: flag poorly fitted samples
│   │   ├── Native_VE_optimizer_all5start.py  # step 3: 5-start re-optimization of flagged samples
│   │   ├── Native_VE_plot.py                 # step 4: fit figure per sample
│   │   └── Submit_VE_native_autosumit.sh     # SLURM job (self-resubmitting chain)
│   └── BPE_IsoFiber/
│       └── Native_BPE_IsoFiber_optimizer_final.py  # single-start + warm-start chaining
└── Data_Example/
    ├── README.md
    ├── FE_model/
    │   ├── VE/VE_45d_template_final.feb
    │   └── BPE_IsoFiber/BPE_IsoFiber_90d_Model.feb
    └── Experimental_data_for_optimization/
        └── Degraded_cartilage_csv_for45dModel/   # 3 example samples + VE results
```

---

## Workflow at a glance

```
experimental CSV (time, displ, load)
        │
        ▼
 setup_loading_protocol()  ── once per sample ──►  loaded.feb
   (measured displacement history → FEBio load curve)
        │
        ▼
 ┌──────────── Nelder–Mead loop ─────────────┐
 │ replace material parameters → run.feb     │
 │ febio4 -i run.feb → rb_force.log (Fz)     │
 │ objective(F_sim, F_exp)                   │
 └───────────────────────────────────────────┘
        │
        ▼
 best parameters, R², R²_hold, RMSE, best_fit_curve.csv
```

Key design points (details in [`docs/optimization_method.md`](docs/optimization_method.md)):

- **Specimen-specific loading** – the FE model is driven by each sample's measured
  displacement history (normalized load curve + peak amplitude). This is written once
  per sample; each iteration only substitutes material parameters. Geometry and mesh
  are never changed during optimization.
- **Objective** – equal-weighted sum of a ramp-phase error, a hold-phase error and a
  peak-force error, all made dimensionless (`W_RAMP = W_HOLD = W_PEAK = 1`).
- **Optimizer** – `scipy.optimize.minimize(method="Nelder-Mead", adaptive=True)` with a
  user-defined initial simplex; parameter bounds enforced by a penalty.
- **Robustness** – failed FEBio runs return a penalty value; evaluations are cached
  and logged; runs can be resumed after a cluster job is pre-empted or times out.

### VE pipeline

1. `Native_VE_singleStart_test.py` – every sample is optimized once from a common
   start point.
2. `Analyze_suspicious.py` – flags samples with a poor or suspicious fit
   (low R² or R²_hold, `E` near its upper bound, or low `g1` with mediocre R²).
3. `Native_VE_optimizer_all5start.py` – flagged samples are re-optimized from five
   fixed start points that cover the expected parameter range; the start with the
   lowest objective is kept.
4. `Native_VE_plot.py` – one figure per sample (fit, residuals, parameters,
   goodness of fit).

### BPE + IsoFiber pipeline

- `Native_BPE_IsoFiber_optimizer_final.py` – one start per sample; samples are processed
  in filename order and the optimum of the previous sample is used as the start point of
  the next (**warm-start chaining**). Parameters are optimized in log10 space. The thesis
  results were obtained in this sequential mode (the optional array mode, with a sample
  index as command-line argument, runs one sample from the default start without chaining).
- Poorly fitted samples were flagged and re-optimized from multiple start points using the
  same logic as the VE pipeline (scripts not included).

The scripts show the configuration for the native group; for the other specimen groups
only the paths, the filename pattern and the parameter bounds were changed.

---

## Requirements

- Python 3 with `numpy`, `pandas`, `scipy`, `matplotlib` (see `requirements.txt`)
- [FEBio 4](https://febio.org/) (`febio4` executable)
- SLURM cluster for the provided job script (optional)

## Related repository

- [CartilageIndentationAnalysis](https://github.com/artorg-unibe-ch/CartilageIndentationAnalysis) –
  processing of the raw indentation data (Hertz and Prony-series fits).

## Author

Keke Xia – MSc thesis, ARTORG Center, University of Bern (2026)
