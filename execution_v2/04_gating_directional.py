"""
Phase 4: Ensemble Confidence Gating + Phase 6: Directional Asymmetry
Tests stricter ensemble rules and LONG-only / asymmetric thresholds.
Uses single model with varying thresholds as proxy for ensemble logic.
"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (load_holdout_with_predictions, load_ticks_after,
                    resolve_trade_exec, compute_expectancy, RESULTS_DIR)

def test_config(sig, tp_mult, sl_mult, label=""):
    """Run trades for a filtered signal set with given R:R."""
    wins, losses, pnl = 0, 0, []
    for idx in range(len(sig)):
        row = sig.iloc[idx]
        ft = load_ticks_after(row["date"], max_ticks=5000)
        if len(ft) < 10: continue
        o, _, _, _ = resolve_trade_exec(
            ft.iloc[0], float(row["brick_size"]),
            bool(row["uptrend"]), ft.iloc[1:], tp_mult=tp_mult, sl_mult=sl_mult)
        if o == 1: wins += 1; pnl.append(float(row["brick_size"]) * tp_mult)
        elif o == 0: losses += 1; pnl.append(-float(row["brick_size"]) * sl_mult)
    total = wins + losses
    if total == 0: return None
    wr = wins / total
    exp = compute_expectancy(wr, tp_mult, sl_mult)
    cum = sum(pnl)
    return {"wr": round(wr, 6), "n": total, "wins": wins, "losses": losses,
            "exp": round(exp, 6), "cum": round(float(cum), 2),
            "avg_pnl": round(float(np.mean(pnl)), 4) if pnl else 0}

def main():
    print("="*70+"\n PHASE 4+6: ENSEMBLE GATING & DIRECTIONAL ASYMMETRY\n"+"="*70)
    df = load_holdout_with_predictions()
    print(f"Loaded {len(df):,} holdout bricks")
    
    tp, sl = 2.0, 1.0  # Use best R:R from Phase 1
    results = {}
    
    # ═══════════════════════════════════════════════════════════
    # ENSEMBLE GATING (threshold sweep)
    # ═══════════════════════════════════════════════════════════
    print("\n THRESHOLD SWEEP (both directions, R:R 1:2.0)")
    print(f"{'Config':<35} {'N':>7} {'WR':>8} {'Exp':>8} {'CumPnL':>10}")
    print("-"*72)
    
    gate_configs = [
        ("PW>=0.5, OS>=1.0", 0.5, 1.0),
        ("PW>=0.5, OS>=1.3", 0.5, 1.3),
        ("PW>=0.5, OS>=1.6", 0.5, 1.6),
        ("PW>=0.5, OS>=1.8", 0.5, 1.8),
        ("PW>=0.5, OS>=2.0", 0.5, 2.0),
        ("PW>=0.6, OS>=1.3", 0.6, 1.3),
        ("PW>=0.6, OS>=1.6", 0.6, 1.6),
        ("PW>=0.7, OS>=1.3", 0.7, 1.3),
        ("PW>=0.7, OS>=1.6", 0.7, 1.6),
        ("PW>=0.7, OS>=1.8", 0.7, 1.8),
        ("PW>=0.8, OS>=1.6", 0.8, 1.6),
        ("PW>=0.8, OS>=1.8", 0.8, 1.8),
    ]
    
    for name, pw_th, os_th in gate_configs:
        sig = df[(df["prob_win"]>=pw_th)&(df["pred_os"]>=os_th)].reset_index(drop=True)
        r = test_config(sig, tp, sl)
        if r:
            print(f"  {name:<33} {r['n']:>7} {r['wr']:>8.4f} {r['exp']:>+8.4f} {r['cum']:>10.1f}")
            results[f"gate_{name}"] = r
    
    # ═══════════════════════════════════════════════════════════
    # DIRECTIONAL ASYMMETRY
    # ═══════════════════════════════════════════════════════════
    print("\n\n DIRECTIONAL ANALYSIS (R:R 1:2.0)")
    print(f"{'Config':<35} {'N':>7} {'WR':>8} {'Exp':>8} {'CumPnL':>10}")
    print("-"*72)
    
    dir_configs = [
        # LONG only with various thresholds
        ("LONG only, PW>=0.5 OS>=1.3", True, None, 0.5, 1.3),
        ("LONG only, PW>=0.5 OS>=1.6", True, None, 0.5, 1.6),
        ("LONG only, PW>=0.7 OS>=1.6", True, None, 0.7, 1.6),
        # SHORT only
        ("SHORT only, PW>=0.5 OS>=1.3", None, True, 0.5, 1.3),
        ("SHORT only, PW>=0.5 OS>=1.6", None, True, 0.5, 1.6),
        ("SHORT only, PW>=0.7 OS>=1.6", None, True, 0.7, 1.6),
        # Asymmetric: lenient LONG + strict SHORT
        ("LONG OS>=1.3 + SHORT OS>=1.8", "asym", None, 0.5, 0),
    ]
    
    for name, long_flag, short_flag, pw_th, os_th in dir_configs:
        if long_flag == "asym":
            # Asymmetric: LONG with OS>=1.3, SHORT with OS>=1.8
            long_sig = df[(df["prob_win"]>=0.5)&(df["pred_os"]>=1.3)&(df["uptrend"]==True)]
            short_sig = df[(df["prob_win"]>=0.5)&(df["pred_os"]>=1.8)&(df["uptrend"]==False)]
            sig = pd.concat([long_sig, short_sig]).sort_values("date").reset_index(drop=True)
        elif long_flag:
            sig = df[(df["prob_win"]>=pw_th)&(df["pred_os"]>=os_th)&(df["uptrend"]==True)].reset_index(drop=True)
        elif short_flag:
            sig = df[(df["prob_win"]>=pw_th)&(df["pred_os"]>=os_th)&(df["uptrend"]==False)].reset_index(drop=True)
        else:
            continue
        
        r = test_config(sig, tp, sl)
        if r:
            print(f"  {name:<33} {r['n']:>7} {r['wr']:>8.4f} {r['exp']:>+8.4f} {r['cum']:>10.1f}")
            results[f"dir_{name}"] = r
    
    # Also test different R:R for LONG-only
    print("\n\n LONG-ONLY R:R SWEEP (PW>=0.5, OS>=1.6)")
    print(f"{'R:R':<12} {'N':>7} {'WR':>8} {'Exp':>8} {'CumPnL':>10}")
    print("-"*50)
    
    long_sig = df[(df["prob_win"]>=0.5)&(df["pred_os"]>=1.6)&(df["uptrend"]==True)].reset_index(drop=True)
    for tp_test in [1.0, 1.5, 2.0, 2.5, 3.0]:
        r = test_config(long_sig, tp_test, 1.0)
        if r:
            print(f"  1:{tp_test:<8} {r['n']:>7} {r['wr']:>8.4f} {r['exp']:>+8.4f} {r['cum']:>10.1f}")
            results[f"long_rr_1:{tp_test}"] = r
    
    with open(RESULTS_DIR / "04_gating_directional.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # Gate configs
    gate_names = [k for k in results if k.startswith("gate_")]
    if gate_names:
        gn = [k.replace("gate_","") for k in gate_names]
        ge = [results[k]["exp"] for k in gate_names]
        gc = ['green' if e>0 else 'red' for e in ge]
        y = np.arange(len(gn))
        axes[0].barh(y, ge, color=gc)
        axes[0].set_yticks(y); axes[0].set_yticklabels(gn, fontsize=8)
        for i, k in enumerate(gate_names):
            axes[0].text(max(ge[i],0)+0.01, i, f'n={results[k]["n"]}', va='center', fontsize=7)
        axes[0].axvline(x=0,color='black',linewidth=1)
        axes[0].set_xlabel('Expectancy');axes[0].set_title('Threshold Gating',fontweight='bold')
        axes[0].grid(True,alpha=0.3,axis='x')
    
    # Directional
    dir_names = [k for k in results if k.startswith("dir_")]
    if dir_names:
        dn = [k.replace("dir_","") for k in dir_names]
        de = [results[k]["exp"] for k in dir_names]
        dc = ['green' if e>0 else 'red' for e in de]
        y = np.arange(len(dn))
        axes[1].barh(y, de, color=dc)
        axes[1].set_yticks(y); axes[1].set_yticklabels(dn, fontsize=7)
        for i, k in enumerate(dir_names):
            axes[1].text(max(de[i],0)+0.01, i, f'n={results[k]["n"]}', va='center', fontsize=7)
        axes[1].axvline(x=0,color='black',linewidth=1)
        axes[1].set_xlabel('Expectancy');axes[1].set_title('Directional Analysis',fontweight='bold')
        axes[1].grid(True,alpha=0.3,axis='x')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_gating_directional.png", dpi=150)
    print(f"\n💾 Saved: 04_gating_directional.json, 04_gating_directional.png")

if __name__ == "__main__": main()
