"""
批量出图：遍历 runs/*/best_fit_curve.csv，配 VE_healthy.csv 的参数，
每个样本存一张 fit 图到 runs/<sample>/fit.png。HPC 无显示，用 Agg。
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")                      # HPC 无 GUI
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

WORKDIR = Path("/storage/homefs/kx24c057/Cartilage/NewExperimentData_for_Optimization/Native_cartilage_csv_for45dModel")
RUN_ROOT = WORKDIR / "runs"
RESULTS_CSV = WORKDIR / "VE_healthy.csv"
MODEL_LABEL = "VE"

results = pd.read_csv(RESULTS_CSV).set_index("sample")
C_EXP, C_SIM, C_RES = "#3E7CB1", "#D1352C", "#7FA8D0"
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.1})

def r2(e, s):
    ss = np.sum((e - e.mean())**2)
    return np.nan if ss <= 0 else 1 - np.sum((s - e)**2) / ss

for curve_path in sorted(RUN_ROOT.glob("*/best_fit_curve.csv")):
    sample = curve_path.parent.name
    if sample not in results.index:
        print(f"skip {sample}: not in results csv"); continue
    summ = results.loc[sample]
    curve = pd.read_csv(curve_path)

    t   = curve["time"].to_numpy(float)
    exp = curve["exp_force"].to_numpy(float) * 1e3      # N -> mN
    sim = curve["sim_force"].to_numpy(float) * 1e3
    res = curve["residual"].to_numpy(float) * 1e6       # N -> µN
    t_peak = float(t[np.argmax(exp)])

    mh = t > t_peak; mr = ~mh
    R2, R2r, R2h = r2(exp, sim), r2(exp[mr], sim[mr]), r2(exp[mh], sim[mh])
    RMSE = np.sqrt(np.mean((sim - exp)**2)) * 1e3       # mN -> µN
    peak_exp, eq_exp = exp.max(), exp[-1]
    peak_sim, eq_sim = sim.max(), sim[-1]

    fig = plt.figure(figsize=(9.6, 8.2))
    gs = GridSpec(3, 1, height_ratios=[3.0, 1.0, 0.9], hspace=0.32)

    ax = fig.add_subplot(gs[0])
    ax.scatter(t, exp, s=10, color=C_EXP, alpha=0.55, edgecolors="none", label="Experiment", zorder=2)
    ax.plot(t, sim, color=C_SIM, lw=2.4, label=MODEL_LABEL, zorder=3)
    ax.axvline(t_peak, ls="--", color="0.6", lw=1.3, label=f"Peak time ({t_peak:.1f} s)")
    ax.set_ylabel("Force [mN]", fontweight="bold")
    ax.set_title(f"{MODEL_LABEL} Fit \u2014 {sample}", fontweight="bold", fontsize=14, pad=12)
    ax.legend(frameon=True, framealpha=0.95, fontsize=11, loc="upper right")
    ax.grid(alpha=0.18); ax.margins(x=0.01)

    axr = fig.add_subplot(gs[1], sharex=ax)
    axr.fill_between(t, res, 0, color=C_RES, alpha=0.55, lw=0)
    axr.axhline(0, color="black", lw=1.0); axr.axvline(t_peak, ls="--", color="0.6", lw=1.3)
    axr.set_ylabel("Residual [\u00b5N]", fontweight="bold"); axr.set_xlabel("Time [s]", fontweight="bold")
    axr.grid(alpha=0.18)

    axb = fig.add_subplot(gs[2]); axb.axis("off")
    mat = (f"$E$ = {summ.E:.4f} MPa      $g_1$ = {summ.g1:.3f}\n"
           f"$\\nu$ = {summ.nu:.2f} (fixed)     $\\tau_1$ = {summ.t1:.3f} s")
    gof = (f"$R^2_{{overall}}$ = {R2:.4f}    $R^2_{{ramp}}$ = {R2r:.4f}    $R^2_{{hold}}$ = {R2h:.4f}\n"
           f"RMSE = {RMSE:.2f} \u00b5N\n"
           f"peak/eq:  exp = {peak_exp/eq_exp:.2f}    sim = {peak_sim/eq_sim:.2f}")
    axb.text(0.01, 0.5, "Material Parameters\n" + mat, va="center", ha="left", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.6", fc="#EAF2FB", ec=C_EXP, lw=1.5))
    axb.text(0.53, 0.5, "Goodness of Fit\n" + gof, va="center", ha="left", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.6", fc="#FDF1E3", ec="#E08A2E", lw=1.5))

    out_png = curve_path.parent / "fit.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")   # HPC 批量用 200 够，500 太慢太大
    plt.close(fig)
    print("saved ->", out_png)