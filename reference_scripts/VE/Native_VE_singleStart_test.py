"""
Healthy + VE 1-Prony — 批量单起点优化 (HPC) — 防重跑版
======================================================
选项 B：单起点跑全量，快速拿结果。之后用分析器标记需要补 5 起点的样本。

★ 防重跑（三层保护，绝不浪费已跑的 5 起点）：
  A. 样本已在 VE_healthy.csv → 跳过（不管之前单起点还是5起点跑的）
  B. 样本有 multistart_summary.csv（之前5起点跑过部分/全部）
     → 直接取已有起点里的最优，写进 csv，不重跑！
  C. 全新样本 → 单起点跑一次

★ 单起点用正常区初值 E=0.0346（数据证明正常区起点都收敛到好解）
  不用 E=0.3 那种极端起点（会掉进 g1≈2 的坏坑）

★ E 上界 1.0，单 Step 模板
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
TEST_MODE = False

WORKDIR    = Path("/storage/homefs/kx24c057/Cartilage/NewExperimentData_for_Optimization/Native_cartilage_csv_for45dModel")
CSV_DIR    = WORKDIR
TEMPLATE   = "/storage/homefs/kx24c057/Cartilage/FEbio_model/VE_45d_template_final.feb"
FEBIO_EXE  = "/storage/homefs/kx24c057/Cartilage/software/febio/bin/febio4"
RESULTS_CSV = WORKDIR / "VE_healthy.csv"

RUN_ROOT   = WORKDIR / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

NU_FIXED = 0.25

BOUNDS = [
    (0.001, 1.0),    # E  [MPa] 放宽边界
    (0.10, 100.0),   # g1
    (0.3,  30.0),    # t1 [s]
]
W_RAMP, W_HOLD, W_PEAK = 1.0, 1.0, 1.0  # equal weighting
N_RAMP, N_HOLD = 40, 40
FAIL_PENALTY  = 1e6
FEBIO_TIMEOUT = 600

# ── 单起点：正常区初值（数据证明正常区都收敛到好解）──────────────
SINGLE_START = np.array([0.0346, 12.9, 1.88])


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


def rerun_and_metrics(loaded_feb, run_feb, rb_log, run_dir, E, g1, t1, exp_df, exp_t, exp_d):
    """用给定参数跑一次 FEBio，存 best_*，返回指标 dict。"""
    text = loaded_feb.read_text(encoding="utf-8")
    text = replace_material_params_VE(text, E, NU_FIXED, g1, t1)
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
    return r2, r2_hold, rmse


# =====================================================================
# 单样本优化（防重跑）
# =====================================================================
def optimize_one_sample_single(csv_path):
    name = Path(csv_path).stem
    run_dir = RUN_ROOT / name
    run_dir.mkdir(exist_ok=True)

    loaded_feb = run_dir / "loaded.feb"
    run_feb    = run_dir / "run.feb"
    rb_log     = run_dir / "rb_force.log"
    summary_csv = run_dir / "multistart_summary.csv"
    history_csv = run_dir / "history_single.csv"

    exp_df = load_experimental_data(csv_path).copy()
    exp_df["load"] -= exp_df["load"].iloc[0]
    exp_t = exp_df["time"].to_numpy(float)
    exp_d = exp_df["displ"].to_numpy(float)
    peak_exp = float(np.max(exp_df["load"]))
    t_hold = float(exp_t[np.argmax(exp_d)])

    setup_loading_protocol(TEMPLATE, loaded_feb, exp_df, N_RAMP, N_HOLD)

    # ★★★ 防重跑保护 B：如果之前 5 起点跑过（summary 存在），直接取最优，不重跑！
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        valid = prev[prev["objective"] < 1e5]
        if len(valid) > 0:
            best = valid.loc[valid["objective"].idxmin()]
            print(f"  {name}: 已有 {len(prev)} 个起点结果（之前5起点跑的），"
                  f"直接取最优 start_id={int(best['start_id'])}，不重跑", flush=True)
            E_opt, g1_opt, t1_opt = float(best["E"]), float(best["g1"]), float(best["t1"])
            # 重跑一次生成 best_fit（快，1次FEBio）
            r2, r2_hold, rmse = rerun_and_metrics(
                loaded_feb, run_feb, rb_log, run_dir,
                E_opt, g1_opt, t1_opt, exp_df, exp_t, exp_d)
            return {
                "sample": name, "batch": name.split("_")[1],
                "E": E_opt, "g1": g1_opt, "t1": t1_opt, "G_inst": 1+g1_opt,
                "nu": NU_FIXED, "objective": float(best["objective"]),
                "R2": r2, "R2_hold": r2_hold, "RMSE": rmse,
                "ramp_time": t_hold, "max_displ": float(np.max(exp_d)), "peak_exp": peak_exp,
                "n_evals": int(prev["n_evals"].sum()) if "n_evals" in prev else -1,
                "method": f"multistart_{len(prev)}pt_reused",
            }

    # ── 全新样本：单起点跑 ──────────────────────────────────────
    if history_csv.exists():
        history_csv.unlink()

    eval_count = [0]
    cache = {}

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
            if rb_log.exists(): rb_log.unlink()
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

    s = np.clip(SINGLE_START, lo_arr, hi_arr)
    t0 = time.time()
    res = minimize(objective, s, method="Nelder-Mead",
                   options={"maxiter": 300, "fatol": 1e-3, "xatol": 1e-2,
                            "adaptive": True, "initial_simplex": make_simplex(s)})
    elapsed = time.time() - t0

    E_opt, g1_opt, t1_opt = res.x
    r2, r2_hold, rmse = rerun_and_metrics(
        loaded_feb, run_feb, rb_log, run_dir,
        E_opt, g1_opt, t1_opt, exp_df, exp_t, exp_d)

    return {
        "sample": name, "batch": name.split("_")[1],
        "E": E_opt, "g1": g1_opt, "t1": t1_opt, "G_inst": 1+g1_opt,
        "nu": NU_FIXED, "objective": float(res.fun),
        "R2": r2, "R2_hold": r2_hold, "RMSE": rmse,
        "ramp_time": t_hold, "max_displ": float(np.max(exp_d)), "peak_exp": peak_exp,
        "n_evals": eval_count[0], "method": "singlestart",
    }


# =====================================================================
# 主流程
# =====================================================================
def main():
    all_csv = sorted(glob.glob(str(CSV_DIR / "Native_*.csv")))
    csv_files = all_csv
    print(f"*** SINGLE-START FULL RUN: {len(csv_files)} samples ***")

    # 防重跑 A：已在 VE_healthy.csv 的样本跳过
    done = set()
    if RESULTS_CSV.exists():
        done = set(pd.read_csv(RESULTS_CSV)["sample"].tolist())
        print(f"  {len(done)} already in VE_healthy.csv, will skip")

    for i, csv in enumerate(csv_files):
        name = Path(csv).stem
        if name in done:
            print(f"[{i+1}/{len(csv_files)}] {name} — skip (already in csv)")
            continue

        print(f"\n[{i+1}/{len(csv_files)}] {name} — optimizing ...", flush=True)
        try:
            result = optimize_one_sample_single(csv)
            pd.DataFrame([result]).to_csv(
                RESULTS_CSV, mode="a",
                header=not RESULTS_CSV.exists(), index=False)
            print(f"   E={result['E']:.4f} g1={result['g1']:.3f} "
                  f"t1={result['t1']:.3f} | R2={result['R2']:.4f} "
                  f"R2_hold={result['R2_hold']:.4f} | {result['method']}", flush=True)
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)

    print(f"\nDone. Results → {RESULTS_CSV}")
    print("接下来运行 analyze_suspicious.py 标记需要补 5 起点的样本")


if __name__ == "__main__":
    main()
