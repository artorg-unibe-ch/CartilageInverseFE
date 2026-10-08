import re
import sys
import time
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import interp1d


# =====================================================================
# 配置
# =====================================================================
TEST_MODE = False                      # ★ True = 只跑 TEST_SAMPLES
TEST_SAMPLES = []
WORKDIR   = Path("/storage/homefs/kx24c057/Cartilage/Native_cartilage_csv_for90d")
CSV_DIR   = WORKDIR
TEMPLATE  = "/storage/homefs/kx24c057/Cartilage/FEbio_model/BPE_IsoFiber_90d_Model.feb"
FEBIO_EXE = "/storage/homefs/kx24c057/Cartilage/software/febio/bin/febio4"

RESULTS_CSV = WORKDIR / "BPE_IsoFiber_healthy.csv"
RUN_ROOT    = WORKDIR / "runs_BPE_IsoFiber"
RUN_ROOT.mkdir(parents=True, exist_ok=True)

N_FIBER_BLOCKS = 1

# 固定参数
NU_FIXED  = 0.25
ALPHA_FIB = 0.0
BETA_FIB  = 2.0
PHI0      = 0.2

# 优化变量：[log10(E), log10(ksi), log10(k)]   ★E 也是 log 尺度
BOUNDS = [
    (-3.0, -0.3),   # log10(E)   -> 10^(-3) .. 10^(-0.3) ≈ 0.001 .. 0.5 boundary set for healthy cartilage is different from degraded cartilage
    (-1.0,  2.0),   # log10(ksi)
    (-5.0, -1.0),   # log10(k)
]
DEFAULT_START = np.array([np.log10(0.02), np.log10(4.0), np.log10(0.012)], dtype=float)
# ≈ [-1.699, 0.602, -1.921]  E=0.02, ksi=4.0, k=0.012 —— 落在 native 拟合簇中心
W_RAMP, W_HOLD, W_PEAK = 1.0, 1.0, 1.0  # equal weighting (same as VE)
N_RAMP, N_HOLD = 40, 40
FAIL_PENALTY   = 1e6
FEBIO_TIMEOUT  = 3600

ARRAY_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else None


class ConfigError(Exception):
    """模板/正则不匹配 —— 立即中止，绝不当成罚分。"""
    pass


# =====================================================================
# 基础函数
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
    n_steps = max(1, int(np.ceil(t_total / dt_step)))
    step_size = t_total / n_steps

    text, n = re.subn(r"<time_steps>\s*\d+\s*</time_steps>",
                      f"<time_steps>{n_steps}</time_steps>", text, count=1)
    if n != 1:
        raise ConfigError("time_steps not matched")
    text, n = re.subn(r"<step_size>\s*[-+0-9.eE]+\s*</step_size>",
                      f"<step_size>{step_size:.8g}</step_size>", text, count=1)
    if n != 1:
        raise ConfigError("step_size not matched")
    text, n = re.subn(r'<value lc="1">\s*[-+0-9.eE]+\s*</value>',
                      f'<value lc="1">{max_displ:.8g}</value>', text, count=1)
    if n != 1:
        raise ConfigError("value lc=1 not matched")
    points_str = "\n".join(f"\t\t\t\t<pt>{tv:.6g},{dv:.6g}</pt>" for tv, dv in lc_points)
    new_points = f"<points>\n{points_str}\n\t\t\t</points>"
    text, n = re.subn(r"<points>.*?</points>", new_points, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise ConfigError("points not matched")
    Path(output_path).write_text(text, encoding="utf-8")
    return max_displ


def replace_material_params_BPE(text, E, ksi, k):
    # -- 基质 E --
    def _repl_solid(m):
        block, nn = re.subn(r"<E>\s*[-+0-9.eE]+\s*</E>", f"<E>{E:.8g}</E>", m.group(0), count=1)
        if nn != 1:
            raise ConfigError("neo-Hookean block found but <E> not matched")
        return block
    text, n = re.subn(r'<solid type="neo-Hookean">.*?</solid>', _repl_solid, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise ConfigError(f'neo-Hookean matched {n} (expected 1)')

    # -- fiber ksi --
    def _repl_fiber(m):
        block, nn = re.subn(r"<ksi>\s*[-+0-9.eE]+\s*</ksi>", f"<ksi>{ksi:.8g}</ksi>", m.group(0), count=1)
        if nn != 1:
            raise ConfigError("fiber-exp-pow block found but <ksi> not matched")
        return block
    text, n = re.subn(r'<fibers type="fiber-exp-pow">.*?</fibers>', _repl_fiber, text, count=0, flags=re.DOTALL)
    if n != N_FIBER_BLOCKS:
        raise ConfigError(f'fiber-exp-pow matched {n} (expected {N_FIBER_BLOCKS})')

    # -- 渗透率 --
    def _repl_perm(m):
        block, nn = re.subn(r"<perm>\s*[-+0-9.eE]+\s*</perm>", f"<perm>{k:.8g}</perm>", m.group(0), count=1)
        if nn != 1:
            raise ConfigError("permeability block found but <perm> not matched")
        return block
    text, n = re.subn(r'<permeability type="perm-const-iso">.*?</permeability>', _repl_perm, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise ConfigError(f'perm-const-iso matched {n} (expected 1)')

    return text


def verify_template_once(template_path):
    text = Path(template_path).read_text(encoding="utf-8")
    out = replace_material_params_BPE(text, 0.123456, 7.654321, 0.00098765)
    if out.count("7.654321") != N_FIBER_BLOCKS:
        raise ConfigError(f"ksi sentinel appears {out.count('7.654321')} times (expected {N_FIBER_BLOCKS})")
    for tag, val in (("E", "0.123456"), ("perm", "0.00098765")):
        if val not in out:
            raise ConfigError(f"sentinel for <{tag}> not found")
    print(f"[check] template OK: E / {N_FIBER_BLOCKS}x ksi / perm all replaced", flush=True)


def parse_rigid_body_force_log(log_path):
    lines = Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    times, fz = [], []
    cur_t = None
    for line in lines:
        line = line.strip()
        if line.startswith("*Time"):
            cur_t = float(line.split("=")[1].strip())
            continue
        if cur_t is not None and line and not line.startswith("*"):
            parts = line.split()
            if len(parts) == 2:
                try:
                    int(parts[0])
                    times.append(cur_t)
                    fz.append(float(parts[1]))
                except ValueError:
                    pass
    if not times:
        raise ValueError("no data in log")
    return pd.DataFrame({"time": np.array(times, float), "force": np.array(fz, float)})


def compute_objective(exp_df, sim_df, t_hold, peak_exp):
    exp_t = exp_df["time"].to_numpy(float)
    exp_f = exp_df["load"].to_numpy(float)
    interp = interp1d(sim_df["time"].to_numpy(float), sim_df["force"].to_numpy(float),
                      kind="linear", bounds_error=False, fill_value="extrapolate")
    sim_on_exp = interp(exp_t)
    eps = 1e-9
    mask_ramp = exp_t < t_hold
    mse_rel_ramp = float(np.mean(((sim_on_exp[mask_ramp] - exp_f[mask_ramp]) / (peak_exp + eps)) ** 2)) if np.any(mask_ramp) else 0.0
    mask_hold = exp_t >= t_hold
    if np.any(mask_hold):
        denom = np.maximum(np.abs(exp_f[mask_hold]), 0.05 * abs(peak_exp) + eps)
        mse_rel_hold = float(np.mean(((sim_on_exp[mask_hold] - exp_f[mask_hold]) / denom) ** 2))
    else:
        mse_rel_hold = 0.0
    peak_sim = float(np.max(sim_on_exp))
    rel_peak = ((peak_sim - peak_exp) / (peak_exp + eps)) ** 2
    return W_RAMP * mse_rel_ramp + W_HOLD * mse_rel_hold + W_PEAK * rel_peak


def compute_metrics(exp_df, sim_df):
    exp_t = exp_df["time"].to_numpy(float)
    exp_f = exp_df["load"].to_numpy(float)
    interp = interp1d(sim_df["time"], sim_df["force"], kind="linear", bounds_error=False, fill_value="extrapolate")
    sim_on_exp = interp(exp_t)
    resid = sim_on_exp - exp_f
    ss_tot = np.sum((exp_f - np.mean(exp_f)) ** 2)
    r2 = np.nan if ss_tot <= 0 else 1 - np.sum(resid ** 2) / ss_tot
    t_fpeak = float(exp_t[np.argmax(exp_f)])
    mh = exp_t > t_fpeak
    ss_hold = np.sum((exp_f[mh] - np.mean(exp_f[mh])) ** 2)
    r2_hold = np.nan if ss_hold <= 0 else 1 - np.sum(resid[mh] ** 2) / ss_hold
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return r2, r2_hold, rmse, sim_on_exp, resid


def run_febio(run_feb, run_dir, rb_log):
    result = subprocess.run([FEBIO_EXE, "-i", str(run_feb)], cwd=run_dir,
                            capture_output=True, text=True, timeout=FEBIO_TIMEOUT)
    if result.returncode != 0:
        (run_dir / "last_febio_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
        raise RuntimeError(f"FEBio failed rc={result.returncode}")
    if not rb_log.exists():
        raise RuntimeError("FEBio finished but rb_force.log not created")


def params_from_row(row):
    # ★ 三个都取 log10（含 E），保证 chain 起点格式与优化变量一致
    return np.array([np.log10(float(row["E"])),
                     np.log10(float(row["ksi"])),
                     np.log10(float(row["k"]))], dtype=float)


# =====================================================================
# 单样本单起点优化
# =====================================================================
def optimize_one_sample(csv_path, init_params, init_source):
    name = Path(csv_path).stem
    run_dir = RUN_ROOT / name
    run_dir.mkdir(exist_ok=True)

    loaded_feb  = run_dir / "loaded.feb"
    run_feb     = run_dir / "run.feb"
    rb_log      = run_dir / "rb_force.log"
    summary_csv = run_dir / "single_start_summary.csv"
    history_csv = run_dir / "history_single_start.csv"
    best_csv    = run_dir / "best_so_far.csv"

    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        if not prev.empty:
            print(f"  {name}: summary exists, skip", flush=True)
            return prev.iloc[-1].to_dict()

    exp_df = load_experimental_data(csv_path).copy()
    exp_df["load"] -= exp_df["load"].iloc[0]
    exp_t = exp_df["time"].to_numpy(float)
    exp_d = exp_df["displ"].to_numpy(float)
    peak_exp = float(np.max(exp_df["load"]))
    t_hold = float(exp_t[np.argmax(exp_d)])

    setup_loading_protocol(TEMPLATE, loaded_feb, exp_df, N_RAMP, N_HOLD)

    lo_arr = np.array([b[0] for b in BOUNDS])
    hi_arr = np.array([b[1] for b in BOUNDS])
    start = np.clip(np.array(init_params, dtype=float), lo_arr, hi_arr)

    eval_count = [0]
    cache = {}
    best_obj = [np.inf]

    # ★ 断点续跑：不再删除 best/history。若上次留了 best_so_far，就从它 warm-start，
    #   并继承已有的 best_obj —— 超时/被 kill 重跑时不丢进度、不从头再来。
    if best_csv.exists():
        try:
            pb = pd.read_csv(best_csv).iloc[-1]
            start = np.clip(np.array([np.log10(float(pb["E"])),
                                      float(pb["log10ksi"]),
                                      float(pb["log10k"])], dtype=float), lo_arr, hi_arr)
            best_obj[0] = float(pb["objective"])
            print(f"  [resume] {name}: warm-start from best_so_far obj={best_obj[0]:.4g}", flush=True)
        except Exception as e:
            print(f"  [resume] best_so_far 读不了({e})，冷启动", flush=True)
    if history_csv.exists():                 # eval 编号接着往下（纯为可读，不影响结果）
        try:
            eval_count[0] = int(pd.read_csv(history_csv)["eval"].max())
        except Exception:
            pass

    print(f"  --- {name} | init from {init_source}: E={10**start[0]:.4g} "
          f"ksi={10**start[1]:.4g} k={10**start[2]:.3e} ---", flush=True)

    def make_simplex(s):
        # ★ 全 log 尺度：E 用固定 log 步长 0.4，不再用线性 0.005
        perturbations = [
            np.array([0.4, 0.0, 0.0]),   # log10(E)
            np.array([0.0, 0.4, 0.0]),   # log10(ksi)
            np.array([0.0, 0.0, 0.5]),   # log10(k)
        ]
        return np.clip(np.vstack([s] + [s + dx for dx in perturbations]), lo_arr, hi_arr)

    def objective(params):
        log10E, log10ksi, log10k = params          # ★ 三个都是 log
        for v, (lo, hi) in zip(params, BOUNDS):
            if not (lo <= v <= hi):
                return FAIL_PENALTY

        E, ksi, k = 10.0 ** log10E, 10.0 ** log10ksi, 10.0 ** log10k   # ★ 转回真实值
        key = (round(log10E, 6), round(log10ksi, 6), round(log10k, 6))
        if key in cache:
            return cache[key]

        eval_count[0] += 1
        t0 = time.time()
        fail_reason = ""

        text = replace_material_params_BPE(loaded_feb.read_text(encoding="utf-8"), E, ksi, k)
        run_feb.write_text(text, encoding="utf-8")
        if rb_log.exists():
            rb_log.unlink()

        try:
            run_febio(run_feb, run_dir, rb_log)
            sim_df = parse_rigid_body_force_log(rb_log)
            sim_df["time"] -= sim_df["time"].iloc[0]
            sim_df["force"] -= sim_df["force"].iloc[0]
            obj = compute_objective(exp_df, sim_df, t_hold, peak_exp)

            if obj < best_obj[0]:
                best_obj[0] = obj
                shutil.copy(run_feb, run_dir / "best_run.feb")
                shutil.copy(rb_log,  run_dir / "best_rb_force.log")
                pd.DataFrame([{
                    "eval": eval_count[0], "E": E, "ksi": ksi, "k": k,
                    "log10E": log10E, "log10ksi": log10ksi, "log10k": log10k,
                    "objective": obj,
                }]).to_csv(best_csv, index=False)

        except subprocess.TimeoutExpired:
            obj, fail_reason = FAIL_PENALTY, "timeout"
        except RuntimeError as e:
            obj, fail_reason = FAIL_PENALTY, f"febio:{e}"
        except ValueError as e:
            obj, fail_reason = FAIL_PENALTY, f"parse:{e}"

        cache[key] = obj
        pd.DataFrame([{
            "eval": eval_count[0], "E": E, "ksi": ksi, "k": k,
            "log10E": log10E, "log10ksi": log10ksi, "log10k": log10k,
            "objective": obj, "fail_reason": fail_reason,
            "time_s": time.time() - t0,
        }]).to_csv(history_csv, mode="a", header=not history_csv.exists(), index=False)
        return obj

    t0 = time.time()
    res = minimize(objective, start, method="Nelder-Mead",
                   options={"maxiter": 300, "fatol": 1e-4, "xatol": 0.05,
                            "adaptive": True, "initial_simplex": make_simplex(start)})
    elapsed = time.time() - t0

    if not best_csv.exists():
        raise RuntimeError(f"{name}: 所有评估都失败了，没有可用的最优解")

    best = pd.read_csv(best_csv).iloc[-1]
    E_opt, ksi_opt, k_opt = float(best["E"]), float(best["ksi"]), float(best["k"])
    log10ksi_opt, log10k_opt = float(best["log10ksi"]), float(best["log10k"])
    objective_final = float(best["objective"])

    sim_df = parse_rigid_body_force_log(run_dir / "best_rb_force.log")
    sim_df["time"]  -= sim_df["time"].iloc[0]
    sim_df["force"] -= sim_df["force"].iloc[0]

    r2, r2_hold, rmse, sim_on_exp, resid = compute_metrics(exp_df, sim_df)
    peak_sim = float(np.max(sim_df["force"]))
    eq_sim   = float(sim_df["force"].iloc[-1])

    pd.DataFrame({
        "time": exp_t, "exp_force": exp_df["load"].to_numpy(float),
        "sim_force": sim_on_exp, "residual": resid,
    }).to_csv(run_dir / "best_fit_curve.csv", index=False)

    result = {
        "sample": name, "batch": name.split("_")[1],
        "init_source": init_source,
        "init_E": float(10 ** start[0]),           # ★ 真实值
        "init_log10ksi": float(start[1]),
        "init_log10k": float(start[2]),
        "E": float(E_opt), "ksi": float(ksi_opt), "k": float(k_opt),
        "log10E": float(np.log10(E_opt)), "log10ksi": float(log10ksi_opt), "log10k": float(log10k_opt),
        "n_fiber_families": N_FIBER_BLOCKS,
        "phi0": PHI0, "nu": NU_FIXED,
        "alpha_fib": ALPHA_FIB, "beta_fib": BETA_FIB,
        "objective": float(objective_final),
        "R2": r2, "R2_hold": r2_hold, "RMSE": rmse,
        "peak_sim": peak_sim, "eq_sim": eq_sim,
        "ratio_sim": peak_sim / eq_sim if eq_sim else np.nan,
        "ramp_time": t_hold, "max_displ": float(np.max(exp_d)), "peak_exp": peak_exp,
        "n_evals": int(eval_count[0]),
        "best_eval": int(best["eval"]),
        "optimizer_success": bool(res.success),
        "total_time_s": elapsed,
    }
    pd.DataFrame([result]).to_csv(summary_csv, index=False)

    print(f"    -> E={E_opt:.4g} ksi={ksi_opt:.4g} k={k_opt:.3e} | "
          f"R2={r2:.4f} R2_hold={r2_hold:.4f} ratio_sim={result['ratio_sim']:.2f} "
          f"(best at eval {result['best_eval']}/{eval_count[0]}, {elapsed/3600:.2f}h)", flush=True)
    return result


# =====================================================================
# 主流程
# =====================================================================
def main():
    verify_template_once(TEMPLATE)

    all_csv = sorted(glob.glob(str(CSV_DIR / "Native_*.csv")))
    if not all_csv:
        raise ConfigError(f"no CSV matched in {CSV_DIR}")

    if TEST_MODE:
        csv_files = [f for f in all_csv if Path(f).stem in TEST_SAMPLES]
        print(f"*** TEST MODE: {len(csv_files)} samples ***")
    else:
        csv_files = all_csv
        print(f"*** FULL RUN: {len(csv_files)} samples ***")

    if ARRAY_INDEX is not None:
        if not (0 <= ARRAY_INDEX < len(csv_files)):
            raise SystemExit(f"array index {ARRAY_INDEX} out of range 0..{len(csv_files)-1}")
        csv_files = [csv_files[ARRAY_INDEX]]
        print(f"*** ARRAY MODE: index {ARRAY_INDEX} -> {Path(csv_files[0]).stem} ***")

    print(f"Results CSV: {RESULTS_CSV}", flush=True)

    done_rows = {}
    if RESULTS_CSV.exists():
        prev = pd.read_csv(RESULTS_CSV)
        done_rows = {str(r["sample"]): r for _, r in prev.iterrows() if pd.notna(r.get("sample"))}
        print(f"  {len(done_rows)} already done, will skip")

    last_params = DEFAULT_START.copy()
    last_source = "DEFAULT_START"

    for i, csv in enumerate(csv_files):
        name = Path(csv).stem
        if name in done_rows:
            last_params = params_from_row(done_rows[name])
            last_source = name
            print(f"[{i+1}/{len(csv_files)}] {name} -- skip", flush=True)
            continue

        print(f"\n{'=' * 70}")
        print(f"[{i+1}/{len(csv_files)}] {name} -- BPE + {N_FIBER_BLOCKS} fibers")
        print(f"{'=' * 70}", flush=True)
        try:
            result = optimize_one_sample(csv, last_params, last_source)
            pd.DataFrame([result]).to_csv(RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)
            last_params = params_from_row(result)
            last_source = name
        except ConfigError:
            raise
        except Exception as e:
            print(f"   FAILED: {e}", flush=True)

    print(f"\nDone. Results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()