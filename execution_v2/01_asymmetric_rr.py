"""
Phase 1: Asymmetric Risk/Reward Engineering

Tests fixed R:R ratios, Pred_OS-scaled R:R, and spread-adjusted SL
using EXECUTION PRICING on holdout tick data.

Scenario A: Fixed R:R (1:1, 1:1.5, 1:2, 1:2.5, 1:3)
Scenario B: Pred_OS-scaled R:R
Scenario C: Spread-adjusted SL
"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (load_holdout_with_predictions, load_ticks_after,
                    resolve_trade_exec, compute_expectancy, breakeven_wr, RESULTS_DIR)

def main():
    print("="*70)
    print(" PHASE 1: ASYMMETRIC RISK/REWARD ENGINEERING")
    print("="*70)
    
    df = load_holdout_with_predictions()
    print(f"Loaded {len(df):,} holdout bricks with predictions")
    
    # Config thresholds
    pw_th = 0.5; os_th = 1.3
    sig = df[(df["prob_win"] >= pw_th) & (df["pred_os"] >= os_th)].reset_index(drop=True)
    print(f"Signal bricks (PW>={pw_th}, OS>={os_th}): {len(sig):,}")
    
    # ═══════════════════════════════════════════════════════════
    # SCENARIO A: Fixed Asymmetric R:R
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(" SCENARIO A: Fixed Asymmetric R:R Ratios")
    print("="*70)
    
    rr_configs = [
        {"name": "1:1.0", "tp": 1.0, "sl": 1.0},
        {"name": "1:1.5", "tp": 1.5, "sl": 1.0},
        {"name": "1:2.0", "tp": 2.0, "sl": 1.0},
        {"name": "1:2.5", "tp": 2.5, "sl": 1.0},
        {"name": "1:3.0", "tp": 3.0, "sl": 1.0},
        {"name": "0.8:1.5", "tp": 1.5, "sl": 0.8},
        {"name": "0.8:2.0", "tp": 2.0, "sl": 0.8},
    ]
    
    scenario_a = {}
    for rr in rr_configs:
        wins, losses, unresolved = 0, 0, 0
        pnl_pts = []  # Track actual P&L in points
        
        for idx in range(len(sig)):
            row = sig.iloc[idx]
            ft = load_ticks_after(row["date"], max_ticks=5000)
            if len(ft) < 10: continue
            
            entry_tick = ft.iloc[0]
            scan_ticks = ft.iloc[1:]
            outcome, _, ep, xp = resolve_trade_exec(
                entry_tick, float(row["brick_size"]),
                bool(row["uptrend"]), scan_ticks,
                tp_mult=rr["tp"], sl_mult=rr["sl"]
            )
            
            if outcome == 1:
                wins += 1
                pnl_pts.append(float(row["brick_size"]) * rr["tp"])
            elif outcome == 0:
                losses += 1
                pnl_pts.append(-float(row["brick_size"]) * rr["sl"])
            else:
                unresolved += 1
        
        total = wins + losses
        if total == 0: continue
        wr = wins / total
        be = breakeven_wr(rr["tp"], rr["sl"])
        exp = compute_expectancy(wr, rr["tp"], rr["sl"])
        pf = (wins * rr["tp"]) / (losses * rr["sl"]) if losses > 0 else float('inf')
        cum_pnl = sum(pnl_pts)
        avg_pnl = np.mean(pnl_pts) if pnl_pts else 0
        
        print(f"\n  {rr['name']:>10}: WR={wr:.4f} ({wins}W/{losses}L/{unresolved}U) "
              f"BE_WR={be:.4f} Exp={exp:+.4f} PF={pf:.2f} Cum={cum_pnl:.1f}pts")
        
        scenario_a[rr["name"]] = {
            "tp_mult": rr["tp"], "sl_mult": rr["sl"],
            "wins": wins, "losses": losses, "unresolved": unresolved,
            "win_rate": round(wr, 6), "breakeven_wr": round(be, 6),
            "expectancy": round(exp, 6), "profit_factor": round(pf, 4),
            "cum_pnl_pts": round(cum_pnl, 2), "avg_pnl_pts": round(avg_pnl, 4),
            "n_resolved": total
        }
    
    # ═══════════════════════════════════════════════════════════
    # SCENARIO B: Pred_OS-Scaled R:R
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(" SCENARIO B: Pred_OS-Scaled R:R")
    print("="*70)
    
    os_tiers = [
        {"name": "OS>=1.8 → 1:2.5", "os_min": 1.8, "os_max": 99, "tp": 2.5, "sl": 1.0},
        {"name": "OS>=1.6 → 1:2.0", "os_min": 1.6, "os_max": 1.8, "tp": 2.0, "sl": 1.0},
        {"name": "OS>=1.3 → 1:1.5", "os_min": 1.3, "os_max": 1.6, "tp": 1.5, "sl": 1.0},
    ]
    
    scenario_b = {"tiers": {}, "combined": {}}
    all_wins_b, all_losses_b, all_pnl_b = 0, 0, []
    
    for tier in os_tiers:
        tier_sig = sig[(sig["pred_os"] >= tier["os_min"]) & (sig["pred_os"] < tier["os_max"])]
        wins, losses, unresolved = 0, 0, 0
        pnl = []
        
        for idx in range(len(tier_sig)):
            row = tier_sig.iloc[idx]
            ft = load_ticks_after(row["date"], max_ticks=5000)
            if len(ft) < 10: continue
            
            outcome, _, _, _ = resolve_trade_exec(
                ft.iloc[0], float(row["brick_size"]),
                bool(row["uptrend"]), ft.iloc[1:],
                tp_mult=tier["tp"], sl_mult=tier["sl"]
            )
            if outcome == 1:
                wins += 1; pnl.append(float(row["brick_size"]) * tier["tp"])
            elif outcome == 0:
                losses += 1; pnl.append(-float(row["brick_size"]) * tier["sl"])
            else: unresolved += 1
        
        total = wins + losses
        if total == 0: continue
        wr = wins / total
        exp = compute_expectancy(wr, tier["tp"], tier["sl"])
        all_wins_b += wins; all_losses_b += losses; all_pnl_b.extend(pnl)
        
        print(f"  {tier['name']:>25}: WR={wr:.4f} ({total} trades) Exp={exp:+.4f}")
        scenario_b["tiers"][tier["name"]] = {
            "win_rate": round(wr, 6), "n": total, "expectancy": round(exp, 6),
            "wins": wins, "losses": losses
        }
    
    if all_wins_b + all_losses_b > 0:
        combined_wr = all_wins_b / (all_wins_b + all_losses_b)
        combined_exp = np.mean(all_pnl_b) if all_pnl_b else 0
        cum = sum(all_pnl_b)
        print(f"\n  COMBINED: WR={combined_wr:.4f} ({all_wins_b+all_losses_b} trades) "
              f"Avg={combined_exp:.4f} Cum={cum:.1f}pts")
        scenario_b["combined"] = {
            "win_rate": round(combined_wr, 6),
            "n": all_wins_b + all_losses_b,
            "avg_pnl": round(float(combined_exp), 4),
            "cum_pnl": round(float(cum), 2)
        }
    
    # ═══════════════════════════════════════════════════════════
    # SCENARIO C: Spread-Adjusted SL
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(" SCENARIO C: Spread-Adjusted SL")
    print("="*70)
    
    scenario_c = {}
    # Compute z_spread for each signal from its micro tensor
    # Use the spread at brick close tick as proxy
    for config_name, z_spread_th, sl_wide, sl_tight, tp_val in [
        ("Wide SL if high spread", 1.5, 1.5, 1.0, 2.0),
        ("Conservative", 1.0, 1.3, 0.8, 1.5),
    ]:
        wins, losses = 0, 0
        pnl = []
        
        for idx in range(len(sig)):
            row = sig.iloc[idx]
            ft = load_ticks_after(row["date"], max_ticks=5000)
            if len(ft) < 10: continue
            
            # Measure spread at entry
            spread = float(ft.iloc[0]["ask"] - ft.iloc[0]["bid"])
            bs = float(row["brick_size"])
            spread_ratio = spread / bs
            
            # Choose SL based on spread
            sl = sl_wide if spread_ratio > (z_spread_th * 0.15) else sl_tight
            
            outcome, _, _, _ = resolve_trade_exec(
                ft.iloc[0], bs, bool(row["uptrend"]), ft.iloc[1:],
                tp_mult=tp_val, sl_mult=sl
            )
            if outcome == 1:
                wins += 1; pnl.append(bs * tp_val)
            elif outcome == 0:
                losses += 1; pnl.append(-bs * sl)
        
        total = wins + losses
        if total == 0: continue
        wr = wins / total
        cum = sum(pnl)
        avg = np.mean(pnl)
        
        print(f"  {config_name}: WR={wr:.4f} ({total}) Cum={cum:.1f}pts Avg={avg:.4f}")
        scenario_c[config_name] = {
            "win_rate": round(wr, 6), "n": total,
            "cum_pnl": round(float(cum), 2), "avg_pnl": round(float(avg), 4)
        }
    
    # ═══════════════════════════════════════════════════════════
    # SAVE & PLOT
    # ═══════════════════════════════════════════════════════════
    results = {"scenario_a": scenario_a, "scenario_b": scenario_b, "scenario_c": scenario_c}
    with open(RESULTS_DIR / "01_asymmetric_rr.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # A1: WR vs R:R ratio
    names = list(scenario_a.keys())
    wrs = [scenario_a[n]["win_rate"] for n in names]
    bes = [scenario_a[n]["breakeven_wr"] for n in names]
    exps = [scenario_a[n]["expectancy"] for n in names]
    
    x = np.arange(len(names))
    axes[0,0].bar(x - 0.15, wrs, 0.3, label='Actual WR', color='steelblue')
    axes[0,0].bar(x + 0.15, bes, 0.3, label='Break-even WR', color='coral')
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(names, rotation=30)
    axes[0,0].set_ylabel('Win Rate')
    axes[0,0].set_title('Win Rate vs Break-Even by R:R Ratio', fontweight='bold')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3, axis='y')
    
    # A2: Expectancy
    colors = ['green' if e > 0 else 'red' for e in exps]
    axes[0,1].bar(names, exps, color=colors)
    axes[0,1].axhline(y=0, color='black', linewidth=1)
    axes[0,1].set_title('Expectancy per Trade (brick units)', fontweight='bold')
    axes[0,1].set_ylabel('Expectancy')
    axes[0,1].tick_params(axis='x', rotation=30)
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # B: Pred_OS tiers
    if scenario_b["tiers"]:
        tn = list(scenario_b["tiers"].keys())
        tw = [scenario_b["tiers"][n]["win_rate"] for n in tn]
        te = [scenario_b["tiers"][n]["expectancy"] for n in tn]
        x2 = np.arange(len(tn))
        axes[1,0].bar(x2 - 0.15, tw, 0.3, label='WR', color='steelblue')
        axes[1,0].bar(x2 + 0.15, te, 0.3, label='Expectancy', color='darkorange')
        axes[1,0].set_xticks(x2); axes[1,0].set_xticklabels(tn, rotation=15, fontsize=8)
        axes[1,0].axhline(y=0, color='black', linewidth=0.5)
        axes[1,0].set_title('Pred_OS-Scaled R:R Performance', fontweight='bold')
        axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative PnL for best configs
    axes[1,1].text(0.5, 0.5, 'See JSON for\ndetailed results',
                   ha='center', va='center', fontsize=14, transform=axes[1,1].transAxes)
    axes[1,1].set_title('Summary', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_asymmetric_rr.png", dpi=150)
    print(f"\n💾 Saved: 01_asymmetric_rr.json, 01_asymmetric_rr.png")

if __name__ == "__main__": main()
