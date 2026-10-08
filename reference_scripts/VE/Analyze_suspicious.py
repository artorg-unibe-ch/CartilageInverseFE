"""
可疑样本分析器 — 标记需要补 5 起点的 healthy 样本
=================================================
读 VE_healthy.csv（单起点结果），按判据标记"可疑"样本。
可疑样本需要用 5 起点重跑（用 Native_VE_optimizer_all5start.py，
它有防重跑，只会跑还没跑过的样本）。

判据（满足任一即可疑）：
  1. R² < 0.90          → 拟合差
  2. R²_hold < 0.85     → hold 段差
  3. E > 0.95           → 接近上界（可能卡边界，需更高 E）
  4. g1 < 2.0 且 R²<0.95 → 可能掉进 g1≈2 坏坑

用法：
  python analyze_suspicious.py
输出：
  屏幕打印分类 + 写 suspicious_samples.txt（需补5起点的样本名单）
"""

import pandas as pd
from pathlib import Path

WORKDIR = Path("/storage/homefs/kx24c057/Cartilage/ClinicalRiboflavinPart2_UV30min_cartilage_csv")
RESULTS_CSV = WORKDIR / "VE_CR2.csv"
OUT_TXT = WORKDIR / "suspicious_samples.txt"

# 判据阈值
R2_MIN      = 0.90
R2_HOLD_MIN = 0.85
E_UPPER     = 0.95
G1_LOW      = 2.0
R2_G1_CHECK = 0.95


def main():
    if not RESULTS_CSV.exists():
        print(f"找不到 {RESULTS_CSV}，先跑单起点优化")
        return

    df = pd.read_csv(RESULTS_CSV)
    print(f"读入 {len(df)} 个样本的结果\n")

    suspicious = []
    reasons = {}

    for _, row in df.iterrows():
        name = row["sample"]
        r2 = row.get("R2", float("nan"))
        r2_hold = row.get("R2_hold", float("nan"))
        E = row.get("E", float("nan"))
        g1 = row.get("g1", float("nan"))

        rs = []
        if r2 < R2_MIN:
            rs.append(f"R²={r2:.3f}<{R2_MIN}")
        if r2_hold < R2_HOLD_MIN:
            rs.append(f"R²_hold={r2_hold:.3f}<{R2_HOLD_MIN}")
        if E > E_UPPER:
            rs.append(f"E={E:.3f}>{E_UPPER}(近上界)")
        if g1 < G1_LOW and r2 < R2_G1_CHECK:
            rs.append(f"g1={g1:.2f}<{G1_LOW}且R²不高(疑坏坑)")

        if rs:
            suspicious.append(name)
            reasons[name] = rs

    # 打印分类
    print("="*70)
    print(f"正常样本（{len(df)-len(suspicious)} 个）— 单起点结果可采用：")
    print("="*70)
    for _, row in df.iterrows():
        if row["sample"] not in suspicious:
            print(f"  ✓ {row['sample']:35s} R²={row['R2']:.3f} "
                  f"R²_hold={row['R2_hold']:.3f} E={row['E']:.4f} g1={row['g1']:.2f}")

    print()
    print("="*70)
    print(f"⚠️  可疑样本（{len(suspicious)} 个）— 需要补 5 起点：")
    print("="*70)
    for name in suspicious:
        row = df[df["sample"] == name].iloc[0]
        print(f"  ⚠️  {name:35s} R²={row['R2']:.3f} "
              f"R²_hold={row['R2_hold']:.3f} E={row['E']:.4f} g1={row['g1']:.2f}")
        print(f"      原因: {', '.join(reasons[name])}")

    # 写名单
    with open(OUT_TXT, "w") as f:
        for name in suspicious:
            f.write(name + "\n")

    print()
    print(f"可疑样本名单 → {OUT_TXT}")
    if suspicious:
        print(f"\n下一步：删掉这些样本在 VE_healthy.csv 里的行，然后用")
        print(f"        Native_VE_optimizer_all5start.py 重跑（会自动只跑这些）")
        print(f"        （或用下面的辅助命令）")
        print(f"\n辅助：从 VE_healthy.csv 移除可疑样本（备份后）：")
        print(f"  python analyze_suspicious.py --remove")
    else:
        print("\n✓ 没有可疑样本，全部单起点结果可用！")


def remove_suspicious():
    """从 VE_healthy.csv 移除可疑样本，让 5 起点脚本重跑它们。"""
    if not OUT_TXT.exists():
        print("先运行分析（不带 --remove）生成名单")
        return
    suspicious = [l.strip() for l in open(OUT_TXT) if l.strip()]
    if not suspicious:
        print("没有可疑样本")
        return
    df = pd.read_csv(RESULTS_CSV)
    # 备份
    backup = RESULTS_CSV.with_suffix(".csv.bak")
    df.to_csv(backup, index=False)
    print(f"已备份到 {backup}")
    # 移除可疑样本的行 + 删它们的 multistart_summary（强制5起点重跑）
    df_keep = df[~df["sample"].isin(suspicious)]
    df_keep.to_csv(RESULTS_CSV, index=False)
    print(f"已从 VE_healthy.csv 移除 {len(suspicious)} 个可疑样本")
    for name in suspicious:
        summ = WORKDIR / "runs" / name / "multistart_summary.csv"
        if summ.exists():
            summ.unlink()
            print(f"  删除 {name}/multistart_summary.csv（强制5起点重跑）")
    print(f"\n现在跑 Native_VE_optimizer_all5start.py，会只对这 {len(suspicious)} 个样本跑5起点")


if __name__ == "__main__":
    import sys
    if "--remove" in sys.argv:
        remove_suspicious()
    else:
        main()