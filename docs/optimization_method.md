# Optimization method

This document describes the logic implemented in `reference_scripts/`. Symbols refer to
variable names in the code.

## 1. Input data

One CSV per sample with columns `time` [s], `displ` [mm], `load` [N]
(FEBio unit system mm–N–s, so `E` is in MPa).

Pre-processing inside the optimizer:

- `load` is baseline-corrected by subtracting its first value.
- `peak_exp = max(load)`.
- `t_hold` = time of maximum displacement (end of ramp / start of hold).

## 2. Loading protocol (once per sample)

`setup_loading_protocol()` turns the FE template into `runs/<sample>/loaded.feb`:

1. The measured displacement is normalized by its maximum `max_displ`.
2. The normalized curve is resampled to 40 points on the ramp (start → displacement
   peak) and 40 points on the hold (`N_RAMP = N_HOLD = 40`); the first point is set
   to (0, 0).
3. These points replace the load-curve `<points>`; `max_displ` becomes the amplitude
   of the rigid indenter's prescribed z-displacement (`<value lc="1">`).
4. `<time_steps>` and `<step_size>` are set for a nominal step of 0.1 s up to the last
   time point (FEBio's auto time stepper is active in the templates).

Geometry, mesh, boundary conditions and contact are taken unchanged from the template.

## 3. Forward evaluation (every iteration)

1. Read `loaded.feb`, substitute the material parameters by regex, write `run.feb`.
2. Run `febio4 -i run.feb` in the sample's run folder.
3. Parse the rigid-body reaction force `Fz` from `rb_force.log`.
4. Shift simulated time and force so that both start at zero.
5. Linearly interpolate the simulated force onto the experimental time points.

| Model | Parameters written into the `.feb` |
|---|---|
| VE | `<E>`, `<v>` in the neo-Hookean `<elastic>` block; `<g1>`, `<t1>` of the viscoelastic material |
| BPE + IsoFiber | `<E>` of the neo-Hookean `<solid>`; `<ksi>` of the `fiber-exp-pow` fibres; `<perm>` of `perm-const-iso` |

The BPE script additionally checks the template once at start-up
(`verify_template_once`): sentinel values are substituted and must be found in the
output (`ksi` exactly once per fibre block). A mismatch raises `ConfigError` and stops the run, so a
template/regex error is never silently turned into a penalty.

In the VE model, `E` is the long-term (equilibrium) modulus of the neo-Hookean
elastic response; the instantaneous modulus scales with `G_inst = 1 + g1`
(reported in the result tables).

## 4. Objective function

With `F_sim` interpolated at the experimental time points, `F_exp` the baseline-corrected
experimental force and `F_peak = peak_exp`:

```
J = W_RAMP · mean_{t <  t_hold} [ (F_sim − F_exp) / F_peak ]²
  + W_HOLD · mean_{t >= t_hold} [ (F_sim − F_exp) / D(t)  ]²
  + W_PEAK · [ (max F_sim − F_peak) / F_peak ]²
```

- `W_RAMP = W_HOLD = W_PEAK = 1` (equal weighting) for both models.
- Hold-phase denominator `D(t)`:
  - VE: `D(t) = F_exp(t)` (pointwise relative error)
  - BPE: `D(t) = max(|F_exp(t)|, 0.05 · F_peak)` (floor avoids blow-up at low forces)
- `max F_sim` is taken over the interpolated simulated curve.

## 5. Optimizer

`scipy.optimize.minimize(method="Nelder-Mead")`, `adaptive=True`, `maxiter=300`.

| | VE | BPE + IsoFiber |
|---|---|---|
| Optimization variables | `E`, `g1`, `t1` (linear) | `log10 E`, `log10 ksi`, `log10 k` |
| Initial simplex (offsets from start) | `E`: +40 %; `g1`: +max(30 %, 0.5); `t1`: +max(50 %, 0.5 s) | +0.4, +0.4, +0.5 decades |
| `fatol` / `xatol` | 1e-3 / 1e-2 | 1e-4 / 0.05 (in log10 units) |

Common details:

- **Bounds**: box bounds are set in the configuration block; a point outside them
  returns `FAIL_PENALTY = 1e6` without running FEBio. Simplex vertices are clipped to
  the bounds.
- **Failed simulations** (non-convergence, timeout, missing/empty log) also return
  `FAIL_PENALTY`.
- **Cache**: parameter vectors rounded to 6 decimals are cached, so repeated vertices
  are not re-simulated.
- **History**: every evaluation is appended to a history CSV (parameters, objective,
  wall time; BPE also logs the failure reason).

## 6. Start points and multi-start strategy

### VE

1. **Single start** for every sample from `[E, g1, t1] = [0.0346, 12.9, 1.88]`
   (the optimum of a typical native sample).
2. **Flagging** (`Analyze_suspicious.py`) – a sample is suspicious if any of:
   - `R² < 0.90`
   - `R²_hold < 0.85`
   - `E > 0.95` (close to the upper bound)
   - `g1 < 2.0` and `R² < 0.95` (possible poor local minimum)
3. **Five-start re-optimization** of the flagged samples
   (`Native_VE_optimizer_all5start.py`) from fixed start points:

   | # | E [MPa] | g1 | t1 [s] | intent |
   |---|---|---|---|---|
   | 1 | 0.0346 | 12.9 | 1.88 | typical native sample |
   | 2 | 0.030 | 13.8 | 1.60 | variant of 1 |
   | 3 | 0.050 | 10.0 | 2.50 | slightly stiffer |
   | 4 | 0.300 | 2.0 | 4.00 | high E / low g1 |
   | 5 | 0.010 | 20.0 | 1.00 | low E / high g1 |

   The start with the lowest objective is kept. `n_starts_converged` counts the starts
   whose objective is within 5 % of the best one.

Workflow: run step 1 → `python Analyze_suspicious.py` →
`python Analyze_suspicious.py --remove` (backs up the results CSV, removes the flagged
rows and their `multistart_summary.csv`) → run step 3, which only processes samples
missing from the results CSV.

### BPE + IsoFiber

- The first sample starts from `E = 0.02 MPa, ksi = 4.0, k = 0.012` (centre of the
  native fits); each following sample (filename order) starts from the optimum of the
  previous one (**warm-start chaining**). The thesis results were obtained this way, with
  all samples processed sequentially in one job. (If the script is called with a sample
  index, e.g. from a SLURM array, only that sample is run, from the default start.)
- Flagged samples were re-optimized from multiple start points with the same logic as
  for VE.

## 7. Resuming interrupted runs

| Level | VE | BPE |
|---|---|---|
| Sample | skipped if already in the results CSV | skipped if in the results CSV or if `single_start_summary.csv` exists |
| Within a sample | 5-start: completed starts stored in `multistart_summary.csv` are skipped; the single-start script reuses an existing `multistart_summary.csv` instead of re-running | restarts from `best_so_far.csv` (warm start, keeps best objective) |

`Submit_VE_native_autosumit.sh` makes the SLURM job self-resubmitting: each job queues
its successor (`--dependency=afterany`), up to `MAX_CHAIN = 20` jobs, and stops once the
results CSV contains `EXPECTED_N` samples.

## 8. Goodness of fit

Computed from the final best simulation:

- `R2` – coefficient of determination over the whole curve.
- `R2_hold` – R² for `t > t(F_exp peak)`, i.e. after the experimental **force** peak
  (note: the objective splits ramp/hold at the **displacement** peak `t_hold`).
- `RMSE` [N].
- The plot script additionally reports `R²_ramp` and the experimental and simulated
  peak/equilibrium force ratio (equilibrium = last data point).

VE: the final simulation is re-run at the optimizer's returned point.
BPE: the best evaluated simulation (`best_run.feb`, `best_rb_force.log`) is used.

## 9. Output files (`runs/<sample>/`)

| File | Content |
|---|---|
| `loaded.feb` | template + sample-specific loading |
| `run.feb`, `rb_force.log`, `run.log`, `run.xplt` | last FEBio evaluation (overwritten each iteration) |
| `best_run.feb`, `best_rb_force.log` | simulation at the optimum |
| `best_fit_curve.csv` | `time, exp_force, sim_force, residual` |
| `history_single.csv` / `history_start<i>.csv` (VE), `history_single_start.csv` (BPE) | all evaluations |
| `multistart_summary.csv` (VE 5-start), `single_start_summary.csv`, `best_so_far.csv` (BPE) | per-start / resume information |
| `fit.png` | figure from `Native_VE_plot.py` |

One row per sample is appended to the group results CSV (e.g. `VE_healthy.csv`).
