"""
Healthy + VE 1-Prony — 批量 5 起点优化 (HPC) — 起点级断点续跑版
================================================================
全部 21 个 native 样本，每个样本跑 5 起点取最优。

★ 5 起点用"方法 A"：固定的覆盖性起点，覆盖 healthy 组的宽参数分布
  （正常样本 E≈0.03/g1≈13，浅快样本如 sample06 E≈0.56/g1≈1，跨度大）
  起点 4 专门覆盖高 E 低 g1 区，确保 sample06 这类样本被抓到。

★ E 上界放宽到 1.0（based on diagA 20260701，sample06 需要 E=0.564）

★ 起点级断点续跑：每个起点完成后写进 runs/<sample>/multistart_summary.csv，
  重跑时跳过已完成的起点，被抢占不浪费。

★ 单 Step 模板（快一倍，无抖动）
"""

import re
import time
import glob
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import shutil

# =====================================================================
# 配置
# =====================================================================
TEST_MODE = False         # ★ 正式跑全量 21 个
TEST_SAMPLES = [
    "Native_20260428_sample14",
    "Native_20260609_sample06",
    "Native_20260519_sample06",
]

WORKDIR    = Path("/storage/homefs/kx24c057/Cartilage/Native_cartilage_csv")
CSV_DIR    = WORKDIR
TEMPLATE   = "/storage/homefs/kx24c057/Cartilage/FEbio_model/VE_45d_template_final.feb"
FEBIO_EXE  = "/storage/homefs/kx24c057/Cartilage/software/febio/bin/febio4"
RESULTS_CSV = WORKDIR / "VE_healthy.csv"

RUN_ROOT   = WORKDIR / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

NU_FIXED = 0.25

BOUNDS = [
    (0.001, 1.0),    # E  [MPa] 放宽边界 based on diagA 20260701
    (0.10, 100.0),   # g1
    (0.3,  30.0),    # t1 [s]
]
W_RAMP, W_HOLD, W_PEAK = 1.0, 1.0, 1.0  # equal weighting (same as single-start)
N_RAMP, N_HOLD = 40, 40
FAIL_PENALTY  = 1e6
FEBIO_TIMEOUT = 600

# ── 5 个覆盖性起点（覆盖 healthy 宽分布）──────────────────────────
START_POINTS = [
    np.array([0.0346, 12.9, 1.88]),   # 1. 正常 healthy 典型（sample14 优化值）
    np.array([0.030,  13.8, 1.60]),   # 2. 正常区变体
    np.array([0.050,  10.0, 2.50]),   # 3. 稍硬正常样本
    np.array([0.300,  2.00, 4.00]),   # 4. 高 E 低 g1（抓 sample06 这类浅快样本）
    np.array([0.010,  20.0, 1.00]),   # 5. 低 E 高 g1（软样本）
]


# =====================================================================
# 函数
# =====================================================================
def load_experimental_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df[["time", "load", "displ"]].dropna().reset_index(drop=True)


def build_normalized_loadcurve(exp_df, n_ramp=40, n_hold=40):
    t = exp_df["time"].to_numpy(float)
    d = exp_df["displ"].to_numpy(float)
    max_displ = float(np.max(d))
    d_norm = d / max_displ
    i_dpeak = int(np.argmax(d))
    idx_ramp = np.linspace(0, i_dpeak, n_ramp, dtype=int)
    idx_hold = np.linspace(i_dpeak, len(t) - 1, n_hold + 1, dtype=int)[1:]
    indices = np.unique(np.concatenate([idx_ramp, idx_hold]))
    points = [(float(t[i]), float(d_norm[i])) for i in indices]
    points[0] = (0.0, 0.0)
    return points, max_displ


def setup_loading_protocol(template_path, output_path, exp_df,
                           n_ramp=40, n_hold=40, dt_step=0.1):
    text = Path(template_path).read_text(encoding="utf-8")
    lc_points, max_displ = build_normalized_loadcurve(exp_df, n_ramp, n_hold)
    t_total = lc_points[-1][0]
    n_steps = int(round(t_total / dt_step))
    text, n = re.subn(r"<time_steps>\s*\d+\s*</time_steps>",
                      f"<time_steps>{n_steps}</time_steps>", text, count=1)
    if n != 1: raise ValueError("time_steps")
    text, n = re.subn(r"<step_size>\s*[-+0-9.eE]+\s*</step_size>",
                      f"<step_size>{dt_step:.6g}</step_size>", text, count=1)
    if n != 1: raise ValueError("step_size")
    text, n = re.subn(r'<value lc="1">\s*[-+0-9.eE]+\s*</value>',
                      f'<value lc="1">{max_displ:.8g}</value>', text, count=1)
    if n != 1: raise ValueError("value")
    points_str = "\n".join(f"\t\t\t\t<pt>{tv:.6g},{dv:.6g}</pt>" for tv, dv in lc_points)
    new_points = f"<points>\n{points_str}\n\t\t\t</points>"
    text, n = re.subn(r"<points>.*?</points>", new_points, text, count=1, flags=re.DOTALL)
    if n != 1: raise ValueError("points")
    Path(output_path).write_text(text, encoding="utf-8")
    return max_displ


def replace_material_params_VE(text, E, nu, g1, t1):
    def _repl_neo(m):
        block = m.group(0)
        block = re.sub(r"<E>\s*[-+0-9.eE]+\s*</E>", f"<E>{E:.8g}</E>", block, count=1)
        block = re.sub(r"<v>\s*[-+0-9.eE]+\s*</v>", f"<v>{nu:.8g}</v>", block, count=1)
        return block
    text, n = re.subn(r'<elastic type="neo-Hookean">.*?</elastic>',
                      _repl_neo, text, count=1, flags=re.DOTALL)
    if n != 1: raise ValueError("neo-Hookean block")
    text, n = re.subn(r"<g1>\s*[-+0-9.eE]+\s*</g1>", f"<g1>{g1:.8g}</g1>", text, count=1)
    if n != 1: raise ValueError("g1")
    text, n = re.subn(r"<t1>\s*[-+0-9.eE]+\s*</t1>", f"<t1>{t1:.8g}</t1>", text, count=1)
    if n != 1: raise ValueError("t1")
    return text


def parse_rigid_body_force_log(log_path):
    lines = Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    times, fz = [], []
    cur_t = None
    for line in lines:
        line = line.strip()
        if line.startswith("*Time"):
            cur_t = float(line.split("=")[1].strip()); continue
        if cur_t is not None and line and not line.startswith("*"):
            parts = line.split()
            if len(parts) == 2:
                try:
                    int(parts[0]); times.append(cur_t); fz.append(float(parts[1]))
                except ValueError:
                    pass
    if not times:
        raise ValueError("no data in log")
    return pd.DataFrame({"time": np.array(times, float), "force": np.array(fz, float)})


def compute_objective(exp_df, sim_df, t_hold, peak_exp):
    exp_t = exp_df["time"].to_numpy(float)
    exp_f = exp_df["load"].to_numpy(float)
    interp = interp1d(sim_df["time"].to_numpy(float),
                      sim_df["force"].to_numpy(float),
                      kind="linear", bounds_error=False, fill_value="extrapolate")
    sim_on_exp = interp(exp_t)
    eps = 1e-9
    mask_ramp = exp_t < t_hold
    if np.any(mask_ramp):
        res_ramp = sim_on_exp[mask_ramp] - exp_f[mask_ramp]
        mse_rel_ramp = float(np.mean((res_ramp / (peak_exp + eps)) ** 2))
    else:
        mse_rel_ramp = 0.0
    mask_hold = exp_t >= t_hold
    if np.any(mask_hold):
        res_hold = sim_on_exp[mask_hold] - exp_f[mask_hold]
        mse_rel_hold = float(np.mean((res_hold / (exp_f[mask_hold] + eps)) ** 2))
    else:
        mse_rel_hold = 0.0
    peak_sim = float(np.max(sim_on_exp))
    rel_peak = ((peak_sim - peak_exp) / (peak_exp + eps)) ** 2
    return W_RAMP * mse_rel_ramp + W_HOLD * mse_rel_hold + W_PEAK * rel_peak


def compute_metrics(exp_df, sim_df):
    exp_t = exp_df["time"].to_numpy(float)
    exp_f = exp_df["load"].to_numpy(float)
    interp = interp1d(sim_df["time"], sim_df["force"], kind="linear",
                      bounds_error=False, fill_value="extrapolate")
    sim_on_exp = interp(exp_t)
    resid = sim_on_exp - exp_f
    r2 = 1 - np.sum(resid**2) / np.sum((exp_f - np.mean(exp_f))**2)
    t_fpeak = float(exp_t[np.argmax(exp_f)])
    mh = exp_t > t_fpeak
    r2_hold = 1 - np.sum(resid[mh]**2) / np.sum((exp_f[mh] - np.mean(exp_f[mh]))**2)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return r2, r2_hold, rmse, sim_on_exp, resid


# =====================================================================
# 单样本 5 起点优化 — 起点级断点续跑
# =====================================================================
def optimize_one_sample_multistart(csv_path):
    name = Path(csv_path).stem
    run_dir = RUN_ROOT / name
    run_dir.mkdir(exist_ok=True)

    loaded_feb = run_dir / "loaded.feb"
    run_feb    = run_dir / "run.feb"
    rb_log     = run_dir / "rb_force.log"
    summary_csv = run_dir / "multistart_summary.csv"

    exp_df = load_experimental_data(csv_path).copy()
    exp_df["load"] -= exp_df["load"].iloc[0]
    exp_t = exp_df["time"].to_numpy(float)
    exp_d = exp_df["displ"].to_numpy(float)
    peak_exp = float(np.max(exp_df["load"]))
    t_hold = float(exp_t[np.argmax(exp_d)])

    setup_loading_protocol(TEMPLATE, loaded_feb, exp_df, N_RAMP, N_HOLD)

    lo_arr = np.array([b[0] for b in BOUNDS])
    hi_arr = np.array([b[1] for b in BOUNDS])

    def make_simplex(start):
        perturbations = [
            np.array([start[0]*0.4, 0.0, 0.0]),
            np.array([0.0, max(start[1]*0.3, 0.5), 0.0]),
            np.array([0.0, 0.0, max(start[2]*0.5, 0.5)]),
        ]
        sim = np.vstack([start] + [start + dx for dx in perturbations])
        return np.clip(sim, lo_arr, hi_arr)

    # ★ 起点级断点续跑
    done_starts = set()
    start_results = []
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        done_starts = set(prev["start_id"].tolist())
        start_results = prev.to_dict("records")
        print(f"  {name}: 已完成起点 {sorted(done_starts)}，将跳过", flush=True)

    t_total0 = time.time()

    for si, start in enumerate(START_POINTS):
        sid = si + 1
        if sid in done_starts:
            print(f"  --- {name} | 起点 {sid}/{len(START_POINTS)}: 已完成，跳过 ---", flush=True)
            continue

        print(f"\n  --- {name} | 起点 {sid}/{len(START_POINTS)}: "
              f"E={start[0]:.4f} g1={start[1]:.1f} t1={start[2]:.1f} ---", flush=True)

        eval_count = [0]
        cache = {}
        history_csv = run_dir / f"history_start{sid}.csv"
        if history_csv.exists():
            history_csv.unlink()

        def objective(params):
            E, g1, t1 = params
            for v, (lo, hi) in zip(params, BOUNDS):
                if not (lo <= v <= hi):
                    return FAIL_PENALTY
            key = (round(E, 6), round(g1, 6), round(t1, 6))
            if key in cache:
                return cache[key]
            eval_count[0] += 1
            t0 = time.time()
            try:
                text = loaded_feb.read_text(encoding="utf-8")
                text = replace_material_params_VE(text, E, NU_FIXED, g1, t1)
                run_feb.write_text(text, encoding="utf-8")
                if rb_log.exists():
                    rb_log.unlink()
                subprocess.run([FEBIO_EXE, "-i", str(run_feb)], cwd=run_dir,
                               capture_output=True, text=True, timeout=FEBIO_TIMEOUT)
                sim_df = parse_rigid_body_force_log(rb_log)
                sim_df["time"]  -= sim_df["time"].iloc[0]
                sim_df["force"] -= sim_df["force"].iloc[0]
                obj = compute_objective(exp_df, sim_df, t_hold, peak_exp)
                cache[key] = obj
                pd.DataFrame([{"eval": eval_count[0], "E": E, "g1": g1, "t1": t1,
                               "G_inst": 1+g1, "objective": obj,
                               "time_s": time.time()-t0}]).to_csv(
                    history_csv, mode="a", header=not history_csv.exists(), index=False)
                return obj
            except Exception:
                cache[key] = FAIL_PENALTY
                pd.DataFrame([{"eval": eval_count[0], "E": E, "g1": g1, "t1": t1,
                               "G_inst": 1+g1, "objective": FAIL_PENALTY,
                               "time_s": time.time()-t0}]).to_csv(
                    history_csv, mode="a", header=not history_csv.exists(), index=False)
                return FAIL_PENALTY

        s = np.clip(start, lo_arr, hi_arr)
        t0 = time.time()
        res = minimize(objective, s, method="Nelder-Mead",
                       options={"maxiter": 300, "fatol": 1e-3, "xatol": 1e-2,
                                "adaptive": True, "initial_simplex": make_simplex(s)})
        elapsed = time.time() - t0

        E_o, g1_o, t1_o = res.x
        text = loaded_feb.read_text(encoding="utf-8")
        text = replace_material_params_VE(text, E_o, NU_FIXED, g1_o, t1_o)
        run_feb.write_text(text, encoding="utf-8")
        if rb_log.exists(): rb_log.unlink()
        subprocess.run([FEBIO_EXE, "-i", str(run_feb)], cwd=run_dir,
                       capture_output=True, text=True, timeout=FEBIO_TIMEOUT)
        try:
            sim_df = parse_rigid_body_force_log(rb_log)
            sim_df["time"]  -= sim_df["time"].iloc[0]
            sim_df["force"] -= sim_df["force"].iloc[0]
            r2, r2_hold, rmse, _, _ = compute_metrics(exp_df, sim_df)
        except Exception:
            r2, r2_hold, rmse = np.nan, np.nan, np.nan

        print(f"    → E={E_o:.4f} g1={g1_o:.3f} t1={t1_o:.3f} | "
              f"R2={r2:.4f} R2_hold={r2_hold:.4f} obj={res.fun:.3e} "
              f"({eval_count[0]} evals, {elapsed:.0f}s)", flush=True)

        start_result = {
            "sample": name, "start_id": sid,
            "init_E": start[0], "init_g1": start[1], "init_t1": start[2],
            "E": E_o, "g1": g1_o, "t1": t1_o, "G_inst": 1+g1_o,
            "R2": r2, "R2_hold": r2_hold, "RMSE": rmse,
            "objective": float(res.fun),
            "n_evals": eval_count[0], "time_s": elapsed,
        }
        start_results.append(start_result)
        pd.DataFrame(start_results).to_csv(summary_csv, index=False)
        print(f"    [起点 {sid} 已保存到 multistart_summary.csv]", flush=True)

    # ── 全部起点完成，找全局最优 ──
    best_row = min(start_results, key=lambda r: r["objective"])
    E_opt, g1_opt, t1_opt = best_row["E"], best_row["g1"], best_row["t1"]

    text = loaded_feb.read_text(encoding="utf-8")
    text = replace_material_params_VE(text, E_opt, NU_FIXED, g1_opt, t1_opt)
    run_feb.write_text(text, encoding="utf-8")
    if rb_log.exists(): rb_log.unlink()
    subprocess.run([FEBIO_EXE, "-i", str(run_feb)], cwd=run_dir,
                   capture_output=True, text=True, timeout=FEBIO_TIMEOUT)
    sim_df = parse_rigid_body_force_log(rb_log)
    sim_df["time"]  -= sim_df["time"].iloc[0]
    sim_df["force"] -= sim_df["force"].iloc[0]
    shutil.copy(run_feb, run_dir / "best_run.feb")
    shutil.copy(rb_log,  run_dir / "best_rb_force.log")
    r2, r2_hold, rmse, sim_on_exp, resid = compute_metrics(exp_df, sim_df)

    pd.DataFrame({
        "time": exp_t, "exp_force": exp_df["load"].to_numpy(float),
        "sim_force": sim_on_exp, "residual": resid,
    }).to_csv(run_dir / "best_fit_curve.csv", index=False)

    objs = [r["objective"] for r in start_results]
    best_obj = min(objs)
    n_near = sum(1 for o in objs if abs(o-best_obj)/(best_obj+1e-12) < 0.05)
    total_elapsed = time.time() - t_total0

    return {
        "sample": name, "batch": name.split("_")[1],
        "E": E_opt, "g1": g1_opt, "t1": t1_opt, "G_inst": 1 + g1_opt,
        "nu": NU_FIXED, "objective": float(best_obj),
        "R2": r2, "R2_hold": r2_hold, "RMSE": rmse,
        "ramp_time": t_hold, "max_displ": float(np.max(exp_d)), "peak_exp": peak_exp,
        "best_start": int(best_row["start_id"]),
        "n_starts_converged": n_near, "total_time_s": total_elapsed,
    }


# =====================================================================
# 主流程
# =====================================================================
def main():
    all_csv = sorted(glob.glob(str(CSV_DIR / "Native_*.csv")))
    if TEST_MODE:
        csv_files = [f for f in all_csv if Path(f).stem in TEST_SAMPLES]
        print(f"*** TEST MODE: {len(csv_files)} samples ***")
    else:
        csv_files = all_csv
        print(f"*** FULL RUN: {len(csv_files)} samples ***")

    print(f"Found {len(all_csv)} native csv files")

    done = set()
    if RESULTS_CSV.exists():
        done = set(pd.read_csv(RESULTS_CSV)["sample"].tolist())
        print(f"  {len(done)} already fully done, will skip")

    for i, csv in enumerate(csv_files):
        name = Path(csv).stem
        if name in done:
            print(f"[{i+1}/{len(csv_files)}] {name} — skip (fully done)")
            continue

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(csv_files)}] {name} — 5-start optimizing ...")
        print(f"{'='*70}", flush=True)
        try:
            result = optimize_one_sample_multistart(csv)
            pd.DataFrame([result]).to_csv(
                RESULTS_CSV, mode="a",
                header=not RESULTS_CSV.exists(), index=False)
            print(f"\n  >>> {name} BEST: E={result['E']:.4f} g1={result['g1']:.3f} "
                  f"t1={result['t1']:.3f} | R2={result['R2']:.4f} "
                  f"R2_hold={result['R2_hold']:.4f} | "
                  f"best_start={result['best_start']} "
                  f"({result['n_starts_converged']}/5 within 5%) "
                  f"{result['total_time_s']/3600:.1f}h", flush=True)
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)

    print(f"\nDone. Results → {RESULTS_CSV}")


if __name__ == "__main__":
    main()