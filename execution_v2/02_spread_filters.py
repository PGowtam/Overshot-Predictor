"""
Phase 2: Spread-Aware Entry Filtering
Tests spread threshold filters combined with asymmetric R:R.
"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (load_holdout_with_predictions, load_ticks_after,
                    resolve_trade_exec, compute_expectancy, RESULTS_DIR)

def main():
    print("="*70+"\n PHASE 2: SPREAD-AWARE ENTRY FILTERING\n"+"="*70)
    df = load_holdout_with_predictions()
    pw_th, os_th = 0.5, 1.3
    sig = df[(df["prob_win"]>=pw_th)&(df["pred_os"]>=os_th)].reset_index(drop=True)
    print(f"Signal bricks: {len(sig):,}")
    
    # Collect spread at entry for all signals
    print("Collecting spread data...")
    spreads, brick_sizes = [], []
    for idx in range(len(sig)):
        ft = load_ticks_after(sig.iloc[idx]["date"], max_ticks=10)
        if len(ft) < 2: spreads.append(None); brick_sizes.append(None); continue
        spreads.append(float(ft.iloc[0]["ask"] - ft.iloc[0]["bid"]))
        brick_sizes.append(float(sig.iloc[idx]["brick_size"]))
    
    sig["spread_at_entry"] = spreads
    sig["entry_brick_size"] = brick_sizes
    sig = sig.dropna(subset=["spread_at_entry"]).reset_index(drop=True)
    sig["spread_pct"] = sig["spread_at_entry"] / sig["entry_brick_size"] * 100
    
    med_spread = sig["spread_at_entry"].median()
    p70_spread = sig["spread_at_entry"].quantile(0.7)
    print(f"Spread stats: median={med_spread:.4f}, P70={p70_spread:.4f}, "
          f"mean_pct={sig['spread_pct'].mean():.1f}%")
    
    # Test R:R = 1:2.0 as base (best from Phase 1 likely)
    tp_mult, sl_mult = 2.0, 1.0
    
    filters = {
        "No filter (baseline)": lambda r: True,
        "spread < median*1.2": lambda r: r["spread_at_entry"] < med_spread * 1.2,
        "spread < median": lambda r: r["spread_at_entry"] < med_spread,
        "spread < P70": lambda r: r["spread_at_entry"] < p70_spread,
        "spread_pct < 15%": lambda r: r["spread_pct"] < 15,
        "spread_pct < 12%": lambda r: r["spread_pct"] < 12,
        "spread_pct < 10%": lambda r: r["spread_pct"] < 10,
        "spread < 0.35": lambda r: r["spread_at_entry"] < 0.35,
        "spread < 0.30": lambda r: r["spread_at_entry"] < 0.30,
    }
    
    results = {}
    print(f"\nTesting filters with R:R = 1:{tp_mult} (SL={sl_mult}x):")
    print(f"\n{'Filter':<28} {'N Sig':>7} {'N Res':>7} {'WR':>8} {'Exp':>8} {'CumPnL':>10}")
    print("-"*75)
    
    for fname, ffunc in filters.items():
        filtered = sig[sig.apply(ffunc, axis=1)].reset_index(drop=True)
        wins, losses, pnl = 0, 0, []
        
        for idx in range(len(filtered)):
            row = filtered.iloc[idx]
            ft = load_ticks_after(row["date"], max_ticks=5000)
            if len(ft) < 10: continue
            o, _, _, _ = resolve_trade_exec(
                ft.iloc[0], float(row["entry_brick_size"]),
                bool(row["uptrend"]), ft.iloc[1:], tp_mult=tp_mult, sl_mult=sl_mult)
            if o == 1: wins += 1; pnl.append(float(row["entry_brick_size"]) * tp_mult)
            elif o == 0: losses += 1; pnl.append(-float(row["entry_brick_size"]) * sl_mult)
        
        total = wins + losses
        if total == 0: continue
        wr = wins / total
        exp = compute_expectancy(wr, tp_mult, sl_mult)
        cum = sum(pnl)
        
        print(f"  {fname:<26} {len(filtered):>7} {total:>7} {wr:>8.4f} {exp:>+8.4f} {cum:>10.1f}")
        results[fname] = {"n_signals": len(filtered), "n_resolved": total,
            "wins": wins, "losses": losses, "wr": round(wr, 6),
            "expectancy": round(exp, 6), "cum_pnl": round(float(cum), 2)}
    
    # Also test with OFI confirmation
    print("\n\nTesting OFI + Spread combo filters (R:R 1:2.0):")
    # We need z_OFI from tensors - approximate from model predictions
    # Higher pred_os correlates with favorable OFI
    combo_filters = {
        "spread<med + OS>=1.6": lambda r: r["spread_at_entry"]<med_spread and r["pred_os"]>=1.6,
        "spread<med + OS>=1.8": lambda r: r["spread_at_entry"]<med_spread and r["pred_os"]>=1.8,
        "spread<0.35 + OS>=1.6": lambda r: r["spread_at_entry"]<0.35 and r["pred_os"]>=1.6,
        "spread<0.35 + OS>=1.8": lambda r: r["spread_at_entry"]<0.35 and r["pred_os"]>=1.8,
    }
    
    print(f"\n{'Filter':<30} {'N Sig':>7} {'N Res':>7} {'WR':>8} {'Exp':>8} {'CumPnL':>10}")
    print("-"*78)
    
    for fname, ffunc in combo_filters.items():
        filtered = sig[sig.apply(ffunc, axis=1)].reset_index(drop=True)
        wins, losses, pnl = 0, 0, []
        for idx in range(len(filtered)):
            row = filtered.iloc[idx]
            ft = load_ticks_after(row["date"], max_ticks=5000)
            if len(ft) < 10: continue
            o, _, _, _ = resolve_trade_exec(
                ft.iloc[0], float(row["entry_brick_size"]),
                bool(row["uptrend"]), ft.iloc[1:], tp_mult=tp_mult, sl_mult=sl_mult)
            if o == 1: wins += 1; pnl.append(float(row["entry_brick_size"]) * tp_mult)
            elif o == 0: losses += 1; pnl.append(-float(row["entry_brick_size"]) * sl_mult)
        total = wins + losses
        if total == 0: continue
        wr = wins / total
        exp = compute_expectancy(wr, tp_mult, sl_mult)
        cum = sum(pnl)
        print(f"  {fname:<28} {len(filtered):>7} {total:>7} {wr:>8.4f} {exp:>+8.4f} {cum:>10.1f}")
        results[fname] = {"n_signals": len(filtered), "n_resolved": total,
            "wins": wins, "losses": losses, "wr": round(wr, 6),
            "expectancy": round(exp, 6), "cum_pnl": round(float(cum), 2)}
    
    with open(RESULTS_DIR / "02_spread_filters.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    names = [n for n in results.keys() if results[n]["n_resolved"] > 5]
    exps = [results[n]["expectancy"] for n in names]
    ns = [results[n]["n_resolved"] for n in names]
    colors = ['green' if e > 0 else 'red' for e in exps]
    y = np.arange(len(names))
    bars = ax.barh(y, exps, color=colors, edgecolor='white')
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Expectancy (brick units)')
    ax.set_title('Spread Filter Effectiveness (R:R 1:2.0)', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    for i, (e, n) in enumerate(zip(exps, ns)):
        ax.text(max(e, 0) + 0.01, i, f'n={n}', va='center', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_spread_filters.png", dpi=150)
    print(f"\n💾 Saved: 02_spread_filters.json, 02_spread_filters.png")

if __name__ == "__main__": main()
