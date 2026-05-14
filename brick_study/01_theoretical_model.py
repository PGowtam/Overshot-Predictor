"""Phase 1: Theoretical Profitability — Price-Proportional Multipliers"""
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SPREAD_MEAN = 0.42
REF_PRICE = 2400  # Approximate 2024 gold price
CURRENT_K = 0.00118
CURRENT_MID_WR = 0.91
CURRENT_EXEC_WR = 0.42

MULTIPLIERS = [0.00118, 0.00150, 0.00177, 0.00200, 0.00236, 0.00295, 0.00354]

def main():
    print("="*70+"\n PHASE 1: THEORETICAL MODEL (PRICE-PROPORTIONAL)\n"+"="*70)
    current_bs = REF_PRICE * CURRENT_K
    deg = CURRENT_MID_WR - CURRENT_EXEC_WR
    deg_per_pct = deg / (SPREAD_MEAN / current_bs)
    rows = []
    for K in MULTIPLIERS:
        bs = REF_PRICE * K
        sp = SPREAD_MEAN / bs
        rt = 2 * sp
        eff_tp = bs - SPREAD_MEAN
        eff_sl = bs + SPREAD_MEAN
        eff_rr = eff_tp / eff_sl
        be_wr = 1 / (1 + eff_rr)
        opt = min(max(CURRENT_MID_WR - deg_per_pct * sp, 0), 1)
        real = min(max(opt - 0.05 * np.log2(K / CURRENT_K), 0), 1)
        pess = min(max(opt - 0.10 * np.log2(K / CURRENT_K), 0), 1)
        rows.append({"K": K, "bs": round(bs, 2), "ratio": round(K/CURRENT_K, 2),
            "spread_pct": round(sp*100, 2), "eff_rr": round(eff_rr, 4),
            "be_wr": round(be_wr*100, 2),
            "opt_wr": round(opt*100, 2), "real_wr": round(real*100, 2), "pess_wr": round(pess*100, 2),
            "opt_margin": round((opt-be_wr)*100, 2),
            "real_margin": round((real-be_wr)*100, 2),
            "pess_margin": round((pess-be_wr)*100, 2)})
    print(f"\n{'K':>8} {'~BS':>6} {'Ratio':>6} {'Spr%':>6} {'BE_WR':>7} "
          f"{'Opt':>7} {'Real':>7} {'Pess':>7} {'OptMar':>8} {'RealMar':>8} {'PessMar':>8}")
    print("-"*92)
    for r in rows:
        flag = "✅" if r["real_margin"]>5 else "⚠️" if r["real_margin"]>0 else "❌"
        print(f"  {flag}{r['K']:>7} {r['bs']:>5.1f} {r['ratio']:>5.1f}x {r['spread_pct']:>5.1f}% "
              f"{r['be_wr']:>6.1f}% {r['opt_wr']:>6.1f}% {r['real_wr']:>6.1f}% {r['pess_wr']:>6.1f}% "
              f"{r['opt_margin']:>+7.1f}% {r['real_margin']:>+7.1f}% {r['pess_margin']:>+7.1f}%")
    with open(RESULTS/"01_theoretical.json","w") as f: json.dump(rows, f, indent=2)
    fig, ax = plt.subplots(figsize=(12, 6))
    ks = [r["K"] for r in rows]
    ax.plot(ks, [r["be_wr"] for r in rows], 'r--', lw=2, label='Break-Even WR')
    ax.plot(ks, [r["opt_wr"] for r in rows], 'g-o', lw=2, label='Optimistic')
    ax.plot(ks, [r["real_wr"] for r in rows], 'b-s', lw=2, label='Realistic')
    ax.plot(ks, [r["pess_wr"] for r in rows], 'm-^', lw=2, label='Pessimistic')
    ax.fill_between(ks, [r["be_wr"] for r in rows], 100, alpha=0.1, color='green')
    ax.set_xlabel('K (brick = price × K)'); ax.set_ylabel('Win Rate (%)')
    ax.set_title('Projected WR vs Break-Even by Multiplier', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS/"01_theoretical.png", dpi=150)
    print(f"\n💾 Saved")

if __name__ == "__main__": main()
