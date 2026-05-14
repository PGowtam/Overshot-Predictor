"""
Phase 7: Monte Carlo Robustness Validation
Validates candidate execution systems under randomized market conditions.
"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import (load_holdout_with_predictions, load_ticks_after, RESULTS_DIR)

N_SIMULATIONS = 1000

def simulate_trade_mc(entry_tick, brick_size, is_long, future_ticks, tp_mult, sl_mult, rng):
    """Single trade with random slippage/spread perturbation."""
    if len(future_ticks) < 5: return None
    
    # Random entry slippage (0-2 ticks offset)
    entry_offset = rng.integers(0, 3)
    if entry_offset >= len(future_ticks) - 5: entry_offset = 0
    tick = future_ticks.iloc[entry_offset]
    
    # Base entry price with random spread perturbation (±20%)
    spread = float(tick["ask"] - tick["bid"])
    spread_perturb = spread * (1 + rng.uniform(-0.2, 0.2))
    mid = (float(tick["bid"]) + float(tick["ask"])) / 2
    
    if is_long:
        entry = mid + spread_perturb / 2
    else:
        entry = mid - spread_perturb / 2
    
    # TP/SL levels
    tp_dist = brick_size * tp_mult
    sl_dist = brick_size * sl_mult
    tp = entry + tp_dist if is_long else entry - tp_dist
    sl = entry - sl_dist if is_long else entry + sl_dist
    
    # Scan with exit slippage
    exit_offset = rng.integers(0, 3)
    scan = future_ticks.iloc[entry_offset + 1:]
    bids, asks = scan["bid"].values, scan["ask"].values
    
    for i in range(len(bids)):
        if is_long:
            # Exit at bid with potential slippage
            ep = bids[i] - rng.uniform(0, spread * 0.1)
            if ep >= tp: return brick_size * tp_mult
            if ep <= sl: return -brick_size * sl_mult
        else:
            ep = asks[i] + rng.uniform(0, spread * 0.1)
            if ep <= tp: return brick_size * tp_mult
            if ep >= sl: return -brick_size * sl_mult
    return None

def run_mc_for_system(sig_df, tp_mult, sl_mult, spread_filter=None, n_sims=N_SIMULATIONS):
    """Run Monte Carlo for a candidate system configuration."""
    # Pre-load all tick data
    print(f"  Pre-loading tick data for {len(sig_df)} signals...")
    tick_data = []
    for idx in range(len(sig_df)):
        ft = load_ticks_after(sig_df.iloc[idx]["date"], max_ticks=500)
        tick_data.append(ft)
    
    sim_results = []
    for sim in range(n_sims):
        rng = np.random.default_rng(42 + sim)
        trades_pnl = []
        
        for idx in range(len(sig_df)):
            row = sig_df.iloc[idx]
            ft = tick_data[idx]
            if len(ft) < 10: continue
            
            # Apply spread filter with noise
            if spread_filter is not None:
                spread = float(ft.iloc[0]["ask"] - ft.iloc[0]["bid"])
                if spread > spread_filter * (1 + rng.uniform(-0.1, 0.1)):
                    continue
            
            pnl = simulate_trade_mc(
                ft.iloc[0], float(row["brick_size"]),
                bool(row["uptrend"]), ft, tp_mult, sl_mult, rng)
            if pnl is not None:
                trades_pnl.append(pnl)
        
        if len(trades_pnl) == 0:
            sim_results.append({"wr": 0, "exp": 0, "cum": 0, "n": 0, "sharpe": 0, "mdd": 0})
            continue
        
        wins = sum(1 for p in trades_pnl if p > 0)
        wr = wins / len(trades_pnl)
        exp = np.mean(trades_pnl)
        cum = sum(trades_pnl)
        
        # Sharpe (annualized, assuming ~250 trades/year)
        if np.std(trades_pnl) > 0:
            sharpe = (np.mean(trades_pnl) / np.std(trades_pnl)) * np.sqrt(min(len(trades_pnl), 250))
        else:
            sharpe = 0
        
        # Max drawdown
        equity = np.cumsum(trades_pnl)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity)
        mdd = np.max(dd) if len(dd) > 0 else 0
        
        sim_results.append({
            "wr": wr, "exp": float(exp), "cum": float(cum),
            "n": len(trades_pnl), "sharpe": float(sharpe), "mdd": float(mdd)
        })
    
    return sim_results

def summarize_mc(sim_results, name):
    """Compute summary statistics from MC simulations."""
    wrs = [s["wr"] for s in sim_results]
    exps = [s["exp"] for s in sim_results]
    cums = [s["cum"] for s in sim_results]
    sharpes = [s["sharpe"] for s in sim_results]
    mdds = [s["mdd"] for s in sim_results]
    ns = [s["n"] for s in sim_results]
    
    prob_profitable = np.mean([1 for c in cums if c > 0]) if cums else 0
    
    summary = {
        "name": name,
        "wr_mean": round(np.mean(wrs), 4),
        "wr_std": round(np.std(wrs), 4),
        "exp_mean": round(np.mean(exps), 4),
        "exp_p5": round(np.percentile(exps, 5), 4),
        "exp_p95": round(np.percentile(exps, 95), 4),
        "cum_mean": round(np.mean(cums), 2),
        "cum_p5": round(np.percentile(cums, 5), 2),
        "cum_p95": round(np.percentile(cums, 95), 2),
        "sharpe_mean": round(np.mean(sharpes), 4),
        "sharpe_p5": round(np.percentile(sharpes, 5), 4),
        "mdd_mean": round(np.mean(mdds), 2),
        "mdd_p95": round(np.percentile(mdds, 95), 2),
        "prob_profitable": round(prob_profitable, 4),
        "trades_mean": round(np.mean(ns), 1),
    }
    return summary

def main():
    print("="*70+"\n PHASE 7: MONTE CARLO ROBUSTNESS VALIDATION\n"+"="*70)
    df = load_holdout_with_predictions()
    print(f"Loaded {len(df):,} holdout bricks")
    
    # Candidate systems to validate
    candidates = [
        {
            "name": "Baseline 1:1",
            "pw": 0.5, "os": 1.3, "tp": 1.0, "sl": 1.0,
            "direction": "both", "spread_filter": None
        },
        {
            "name": "Asymm 1:2 OS>=1.3",
            "pw": 0.5, "os": 1.3, "tp": 2.0, "sl": 1.0,
            "direction": "both", "spread_filter": None
        },
        {
            "name": "Asymm 1:2 OS>=1.6",
            "pw": 0.5, "os": 1.6, "tp": 2.0, "sl": 1.0,
            "direction": "both", "spread_filter": None
        },
        {
            "name": "Asymm 1:2.5 OS>=1.6",
            "pw": 0.5, "os": 1.6, "tp": 2.5, "sl": 1.0,
            "direction": "both", "spread_filter": None
        },
        {
            "name": "Asymm 1:2 OS>=1.6 spread<0.35",
            "pw": 0.5, "os": 1.6, "tp": 2.0, "sl": 1.0,
            "direction": "both", "spread_filter": 0.35
        },
        {
            "name": "LONG 1:2 OS>=1.6",
            "pw": 0.5, "os": 1.6, "tp": 2.0, "sl": 1.0,
            "direction": "long", "spread_filter": None
        },
        {
            "name": "LONG 1:2 OS>=1.6 spread<0.35",
            "pw": 0.5, "os": 1.6, "tp": 2.0, "sl": 1.0,
            "direction": "long", "spread_filter": 0.35
        },
        {
            "name": "Asymm 1:2 PW>=0.7 OS>=1.6",
            "pw": 0.7, "os": 1.6, "tp": 2.0, "sl": 1.0,
            "direction": "both", "spread_filter": None
        },
    ]
    
    all_summaries = {}
    all_sims = {}
    
    for cand in candidates:
        print(f"\n{'='*50}")
        print(f"  System: {cand['name']}")
        print(f"{'='*50}")
        
        # Filter signals
        mask = (df["prob_win"] >= cand["pw"]) & (df["pred_os"] >= cand["os"])
        if cand["direction"] == "long":
            mask &= df["uptrend"] == True
        elif cand["direction"] == "short":
            mask &= df["uptrend"] == False
        sig = df[mask].reset_index(drop=True)
        print(f"  Signals: {len(sig)}")
        
        if len(sig) < 5:
            print("  ⚠️ Too few signals, skipping")
            continue
        
        sims = run_mc_for_system(sig, cand["tp"], cand["sl"], cand["spread_filter"])
        summary = summarize_mc(sims, cand["name"])
        
        print(f"  Results ({N_SIMULATIONS} sims):")
        print(f"    WR: {summary['wr_mean']:.4f} ± {summary['wr_std']:.4f}")
        print(f"    Exp: {summary['exp_mean']:+.4f} [P5={summary['exp_p5']:+.4f}, P95={summary['exp_p95']:+.4f}]")
        print(f"    Cum: {summary['cum_mean']:+.1f} [P5={summary['cum_p5']:+.1f}, P95={summary['cum_p95']:+.1f}]")
        print(f"    Sharpe: {summary['sharpe_mean']:.4f} [P5={summary['sharpe_p5']:.4f}]")
        print(f"    MDD: {summary['mdd_mean']:.1f} [P95={summary['mdd_p95']:.1f}]")
        print(f"    P(profitable): {summary['prob_profitable']:.2%}")
        print(f"    Trades/sim: {summary['trades_mean']:.0f}")
        
        all_summaries[cand["name"]] = summary
        all_sims[cand["name"]] = [s["cum"] for s in sims]
    
    # ═══════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*90)
    print(" MONTE CARLO COMPARISON TABLE")
    print("="*90)
    print(f"{'System':<35} {'WR':>6} {'Exp':>7} {'Sharpe':>7} {'P(Profit)':>10} {'Trades':>7} {'MDD':>7}")
    print("-"*85)
    
    for name, s in all_summaries.items():
        flag = "✅" if s["exp_mean"] > 0 and s["prob_profitable"] > 0.6 else "❌"
        print(f"  {flag} {name:<32} {s['wr_mean']:>6.3f} {s['exp_mean']:>+7.4f} "
              f"{s['sharpe_mean']:>7.3f} {s['prob_profitable']:>10.1%} {s['trades_mean']:>7.0f} {s['mdd_mean']:>7.1f}")
    
    with open(RESULTS_DIR / "07_monte_carlo.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Expectancy distribution
    for name in list(all_sims.keys())[:6]:
        axes[0,0].hist(all_sims[name], bins=30, alpha=0.4, label=name[:25], density=True)
    axes[0,0].axvline(x=0, color='red', linewidth=2)
    axes[0,0].set_xlabel('Cumulative PnL (pts)');axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('MC Cumulative PnL Distribution',fontweight='bold')
    axes[0,0].legend(fontsize=7);axes[0,0].grid(True,alpha=0.3)
    
    # Expectancy comparison
    names = list(all_summaries.keys())
    exps = [all_summaries[n]["exp_mean"] for n in names]
    colors = ['green' if e>0 else 'red' for e in exps]
    y = np.arange(len(names))
    axes[0,1].barh(y, exps, color=colors)
    axes[0,1].set_yticks(y);axes[0,1].set_yticklabels([n[:30] for n in names],fontsize=7)
    axes[0,1].axvline(x=0,color='black',linewidth=1)
    axes[0,1].set_xlabel('Mean Expectancy');axes[0,1].set_title('MC Mean Expectancy',fontweight='bold')
    axes[0,1].grid(True,alpha=0.3,axis='x')
    
    # P(profitable)
    probs = [all_summaries[n]["prob_profitable"] for n in names]
    axes[1,0].barh(y, probs, color=['green' if p>0.6 else 'orange' if p>0.5 else 'red' for p in probs])
    axes[1,0].set_yticks(y);axes[1,0].set_yticklabels([n[:30] for n in names],fontsize=7)
    axes[1,0].axvline(x=0.5,color='red',linestyle='--',alpha=0.5)
    axes[1,0].set_xlabel('P(Profitable)');axes[1,0].set_title('Probability of Profit',fontweight='bold')
    axes[1,0].grid(True,alpha=0.3,axis='x')
    
    # Sharpe
    sharpes = [all_summaries[n]["sharpe_mean"] for n in names]
    axes[1,1].barh(y, sharpes, color=['green' if s>1 else 'orange' if s>0 else 'red' for s in sharpes])
    axes[1,1].set_yticks(y);axes[1,1].set_yticklabels([n[:30] for n in names],fontsize=7)
    axes[1,1].axvline(x=1.0,color='green',linestyle='--',alpha=0.5,label='Sharpe=1')
    axes[1,1].axvline(x=0,color='red',linestyle='--',alpha=0.5)
    axes[1,1].set_xlabel('Sharpe Ratio');axes[1,1].set_title('Risk-Adjusted Return',fontweight='bold')
    axes[1,1].legend();axes[1,1].grid(True,alpha=0.3,axis='x')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "07_monte_carlo.png", dpi=150)
    print(f"\n💾 Saved: 07_monte_carlo.json, 07_monte_carlo.png")

if __name__ == "__main__": main()
