# Data_Example

Minimal example of the inputs and outputs of the inverse FE optimization.

## FE_model/

FEBio 4 templates (units mm–N–s). Geometry, mesh, boundary conditions and contact are
fixed; the optimizer only rewrites the loading protocol (once per sample) and the
material parameters (every iteration).

| File | Model | Used by |
|---|---|---|
| `VE/VE_45d_template_final.feb` | viscoelastic (neo-Hookean + 1-term Prony) | `reference_scripts/VE/` |
| `BPE_IsoFiber/BPE_IsoFiber_90d_Model.feb` | biphasic + isotropic continuous fibre distribution | `reference_scripts/BPE_IsoFiber/` |

## Experimental_data_for_optimization/Degraded_cartilage_csv_for45dModel/

Three degraded-cartilage samples fitted with the **VE** model (single-start step).

| File | Content |
|---|---|
| `Degraded_<date>_<sample>.csv` | input: `time` [s], `displ` [mm], `load` [N] |
| `VE_degraded.csv` | results table, one row per sample (parameters, objective, R², R²_hold, RMSE, number of evaluations, method) |
| `runs/<sample>/` | per-sample outputs, see [`docs/optimization_method.md`](../docs/optimization_method.md#9-output-files-runssample) |

Example results (`VE_degraded.csv`):

| Sample | E [MPa] | g1 | t1 [s] | R² | R²_hold |
|---|---|---|---|---|---|
| SampleB_49 | 0.0094 | 10.94 | 1.75 | 0.993 | 0.990 |
| SampleB_50 | 0.0085 | 6.11 | 3.01 | 0.975 | 0.963 |
| SampleB_51 | 0.0059 | 15.43 | 1.05 | 0.674 | 0.795 |

SampleB_51 is an example of a sample that `Analyze_suspicious.py` flags
(R² < 0.90, R²_hold < 0.85) for five-start re-optimization.

Large intermediate files (`run.xplt`, `run.feb`, `loaded.feb`) are excluded via
`.gitignore`.
