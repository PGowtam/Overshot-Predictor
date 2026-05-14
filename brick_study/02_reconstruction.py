"""
Phase 2+3: Brick Size Feasibility — Price-Proportional Multipliers

Current system: brick_size = price * 0.00236 / 2 = price * 0.00118
Test multipliers: 0.00118 (1x), 0.00150, 0.00177, 0.00200, 0.00236, 0.00295, 0.00354

This keeps brick size dynamic and volatility-adaptive across price regimes.
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

BASE = Path(__file__).resolve().parent.parent
TICK_DIR = BASE / "Data" / "Raw" / "Ticks"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
SPREAD_MEAN = 0.42

# Multipliers to test (K where brick_size = day_open * K)
MULTIPLIERS = [
    (0.00118, "1.0x (current)"),
    (0.00150, "1.27x"),
    (0.00177, "1.5x"),
    (0.00200, "1.69x"),
    (0.00236, "2.0x"),
    (0.00295, "2.5x"),
    (0.00354, "3.0x"),
]

def load_all_2024_ticks():
    print("Loading 2024 tick data...")
    frames = []
    for month in range(1, 13):
        for day in range(1, 32):
            try: d = pd.Timestamp(2024, month, day, tz="UTC")
            except: continue
            p = TICK_DIR/str(d.year)/f"{d.month:02d}"/f"{d.day:02d}.parquet"
            if not p.exists(): continue
            df = pd.read_parquet(p)
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            frames.append(df)
    tdf = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(tdf):,} ticks")
    return tdf

def build_renko_dynamic(bids, asks, timestamps, dates, multiplier):
    """Build Renko with daily-dynamic brick size = day_open * multiplier.
    
    Groups ticks by date. For each day, computes brick_size from day's opening price.
    Carries brick state across days (just like the real system).
    """
    bricks = []
    current = bids[0]
    uptrend = 0
    
    # Group by date for dynamic brick sizing
    unique_dates = sorted(set(dates))
    day_open_prices = {}
    for ud in unique_dates:
        mask = dates == ud
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            day_open_prices[ud] = bids[idxs[0]]
    
    current_bs = bids[0] * multiplier  # Initial
    
    for i in range(1, len(bids)):
        p = bids[i]
        d = dates[i]
        
        # Update brick size at day boundary
        if d in day_open_prices:
            new_bs = day_open_prices[d] * multiplier
            if new_bs > 0:
                current_bs = new_bs
        
        bs = current_bs
        
        if uptrend == 0:
            if p >= current + bs:
                while p >= current + bs:
                    current += bs
                    bricks.append({"close": current, "dir": 1, "size": bs,
                                   "ts": timestamps[i], "tick_idx": i,
                                   "spread": asks[i] - bids[i]})
                uptrend = 1
            elif p <= current - bs:
                while p <= current - bs:
                    current -= bs
                    bricks.append({"close": current, "dir": -1, "size": bs,
                                   "ts": timestamps[i], "tick_idx": i,
                                   "spread": asks[i] - bids[i]})
                uptrend = -1
        elif uptrend == 1:
            if p >= current + bs:
                while p >= current + bs:
                    current += bs
                    bricks.append({"close": current, "dir": 1, "size": bs,
                                   "ts": timestamps[i], "tick_idx": i,
                                   "spread": asks[i] - bids[i]})
            elif p <= current - 2 * bs:
                current -= 2 * bs
                bricks.append({"close": current, "dir": -1, "size": bs,
                                "ts": timestamps[i], "tick_idx": i,
                                "spread": asks[i] - bids[i]})
                uptrend = -1
                while p <= current - bs:
                    current -= bs
                    bricks.append({"close": current, "dir": -1, "size": bs,
                                   "ts": timestamps[i], "tick_idx": i,
                                   "spread": asks[i] - bids[i]})
        else:  # downtrend
            if p <= current - bs:
                while p <= current - bs:
                    current -= bs
                    bricks.append({"close": current, "dir": -1, "size": bs,
                                   "ts": timestamps[i], "tick_idx": i,
                                   "spread": asks[i] - bids[i]})
            elif p >= current + 2 * bs:
                current += 2 * bs
                bricks.append({"close": current, "dir": 1, "size": bs,
                                "ts": timestamps[i], "tick_idx": i,
                                "spread": asks[i] - bids[i]})
                uptrend = 1
                while p >= current + bs:
                    current += bs
                    bricks.append({"close": current, "dir": 1, "size": bs,
                                   "ts": timestamps[i], "tick_idx": i,
                                   "spread": asks[i] - bids[i]})
    return bricks

def main():
    print("="*70+"\n BRICK SIZE FEASIBILITY: PRICE-PROPORTIONAL MULTIPLIERS\n"+"="*70)
    
    tdf = load_all_2024_ticks()
    bids = tdf["bid"].values.astype(np.float64)
    asks = tdf["ask"].values.astype(np.float64)
    ts = tdf["timestamp"].values
    dates = tdf["timestamp"].dt.date.values
    n_days = len(set(dates))
    
    results = {}
    
    for mult, label in MULTIPLIERS:
        # Example brick at ~$2400
        example_bs = 2400 * mult
        print(f"\n{'='*60}")
        print(f"  K={mult} ({label}) → ~{example_bs:.1f} pts at $2400")
        print(f"{'='*60}")
        
        bricks = build_renko_dynamic(bids, asks, ts, dates, mult)
        n_b = len(bricks)
        if n_b < 20:
            print(f"  Only {n_b} bricks — skipping")
            continue
        
        bpd = n_b / max(n_days, 1)
        
        # Brick size stats (dynamic — varies daily)
        brick_sizes = np.array([b["size"] for b in bricks])
        spreads_at_close = np.array([b["spread"] for b in bricks])
        spread_pcts = spreads_at_close / brick_sizes * 100
        
        # Duration
        durs = []
        for i in range(1, len(bricks)):
            d = (bricks[i]["ts"] - bricks[i-1]["ts"]).astype("timedelta64[s]").astype(float)
            if d > 0: durs.append(d)
        mean_dur = np.mean(durs) if durs else 0
        med_dur = np.median(durs) if durs else 0
        
        # Trend persistence
        runs = []; cr = 1
        for i in range(1, len(bricks)):
            if bricks[i]["dir"] == bricks[i-1]["dir"]: cr += 1
            else: runs.append(cr); cr = 1
        runs.append(cr)
        
        # Ticks per brick
        tpb = [bricks[i]["tick_idx"] - bricks[i-1]["tick_idx"] for i in range(1, len(bricks))]
        mean_tpb = np.mean(tpb) if tpb else 0
        buf_cov = (100 / max(mean_tpb, 1)) * (mean_dur / 60)
        
        print(f"  Bricks: {n_b:,} | B/day: {bpd:.1f}")
        print(f"  Brick size: mean={brick_sizes.mean():.2f} std={brick_sizes.std():.2f}")
        print(f"  Spread at close: mean={spreads_at_close.mean():.3f} → {spread_pcts.mean():.1f}% of brick")
        print(f"  Duration: mean={mean_dur/60:.1f}min median={med_dur/60:.1f}min")
        print(f"  Persistence: {np.mean(runs):.1f} | Ticks/brick: {mean_tpb:.0f}")
        print(f"  Buffer coverage: {buf_cov:.1f}min")
        
        # === NAIVE CONTINUATION (execution-priced) ===
        print("  Computing continuation rates...")
        cont = {f"1:{r}": {"w": 0, "l": 0, "u": 0} for r in ["1.0", "1.5", "2.0"]}
        
        for i in range(len(bricks) - 1):
            b = bricks[i]
            bi = b["tick_idx"]
            bs = b["size"]
            is_long = b["dir"] == 1
            if bi + 1 >= len(bids): continue
            
            ep = asks[bi + 1] if is_long else bids[bi + 1]
            end_idx = min(bi + 15001, len(bids))
            
            for rr_name, tp_m, sl_m in [("1:1.0", 1.0, 1.0), ("1:1.5", 1.5, 1.0), ("1:2.0", 2.0, 1.0)]:
                tp = ep + bs * tp_m if is_long else ep - bs * tp_m
                sl = ep - bs * sl_m if is_long else ep + bs * sl_m
                resolved = False
                for j in range(bi + 2, end_idx):
                    xp = bids[j] if is_long else asks[j]
                    if is_long:
                        if xp >= tp: cont[rr_name]["w"] += 1; resolved = True; break
                        if xp <= sl: cont[rr_name]["l"] += 1; resolved = True; break
                    else:
                        if xp <= tp: cont[rr_name]["w"] += 1; resolved = True; break
                        if xp >= sl: cont[rr_name]["l"] += 1; resolved = True; break
                if not resolved: cont[rr_name]["u"] += 1
        
        print(f"  {'R:R':>6} {'N':>7} {'WR':>8} {'Unres':>6}")
        cont_results = {}
        for rr, d in cont.items():
            total = d["w"] + d["l"]
            wr = d["w"] / total if total > 0 else 0
            print(f"  {rr:>6} {total:>7} {wr:>8.4f} {d['u']:>6}")
            cont_results[rr] = {"n": total, "wr": round(wr, 6), "unresolved": d["u"]}
        
        # Break-even and margin calculations
        mean_bs = float(brick_sizes.mean())
        mean_spr = float(spreads_at_close.mean())
        eff_tp = mean_bs - mean_spr
        eff_sl = mean_bs + mean_spr
        eff_rr = eff_tp / eff_sl
        be_wr = 1 / (1 + eff_rr)
        naive_11_wr = cont_results["1:1.0"]["wr"]
        margin = naive_11_wr - be_wr
        
        sig_day = bpd * 0.25
        ann_trades = sig_day * 250
        
        print(f"\n  Break-even WR: {be_wr:.4f} | Naive 1:1 WR: {naive_11_wr:.4f} | "
              f"Margin: {margin:+.4f}")
        print(f"  Projected annual signals (25% filter): {ann_trades:.0f}")
        
        results[f"K={mult}"] = {
            "multiplier": mult, "label": label,
            "example_bs_at_2400": round(example_bs, 2),
            "n_bricks": n_b, "bricks_per_day": round(bpd, 2),
            "brick_size_mean": round(mean_bs, 3), "brick_size_std": round(float(brick_sizes.std()), 3),
            "spread_pct_mean": round(float(spread_pcts.mean()), 2),
            "spread_pct_median": round(float(np.median(spread_pcts)), 2),
            "mean_dur_min": round(mean_dur / 60, 1), "median_dur_min": round(med_dur / 60, 1),
            "trend_persistence": round(float(np.mean(runs)), 2),
            "ticks_per_brick": round(mean_tpb, 0),
            "buffer_coverage_min": round(buf_cov, 1),
            "continuation": cont_results,
            "eff_rr": round(eff_rr, 4), "breakeven_wr": round(be_wr, 4),
            "naive_11_wr": round(naive_11_wr, 4), "margin": round(margin, 4),
            "annual_signals": round(ann_trades, 0)
        }
    
    # ═══════════════════════════════════════════════════════════
    # DECISION MATRIX
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*100)
    print(" DECISION MATRIX: PRICE-PROPORTIONAL BRICK SIZE")
    print("="*100)
    print(f"{'K':>8} {'Label':>12} {'~BS':>6} {'Spr%':>6} {'B/Day':>6} {'1:1WR':>7} "
          f"{'BE_WR':>7} {'Margin':>8} {'AnnSig':>7} {'Persist':>8} {'BufCov':>7}")
    print("-"*97)
    
    for key, r in results.items():
        flag = "✅" if r["margin"] > 0.05 else "⚠️" if r["margin"] > 0 else "❌"
        print(f"  {flag}{r['multiplier']:>7} {r['label']:>12} {r['example_bs_at_2400']:>5.1f} "
              f"{r['spread_pct_mean']:>5.1f}% {r['bricks_per_day']:>6.1f} "
              f"{r['naive_11_wr']*100:>6.2f}% {r['breakeven_wr']*100:>6.2f}% "
              f"{r['margin']*100:>+7.2f}% {r['annual_signals']:>7.0f} "
              f"{r['trend_persistence']:>8.1f} {r['buffer_coverage_min']:>6.1f}m")
    
    with open(RESULTS / "02_brick_multipliers.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    mults = [r["multiplier"] for r in results.values()]
    labels_x = [f"K={m}" for m in mults]
    
    # 1: Continuation WR by R:R
    wr11 = [r["continuation"]["1:1.0"]["wr"]*100 for r in results.values()]
    wr15 = [r["continuation"]["1:1.5"]["wr"]*100 for r in results.values()]
    wr20 = [r["continuation"]["1:2.0"]["wr"]*100 for r in results.values()]
    be = [r["breakeven_wr"]*100 for r in results.values()]
    axes[0,0].plot(mults, wr11, 'o-', lw=2, label='1:1.0 WR')
    axes[0,0].plot(mults, be, 'r--', lw=2, label='Break-even WR')
    axes[0,0].fill_between(mults, be, wr11, where=[w>b for w,b in zip(wr11,be)],
                           alpha=0.2, color='green', label='Profitable zone')
    axes[0,0].set_xlabel('K multiplier'); axes[0,0].set_ylabel('Win Rate (%)')
    axes[0,0].set_title('Naive 1:1 WR vs Break-Even', fontweight='bold')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
    
    # 2: Margin
    margins = [r["margin"]*100 for r in results.values()]
    colors = ['green' if m>5 else 'orange' if m>0 else 'red' for m in margins]
    axes[0,1].bar(labels_x, margins, color=colors)
    axes[0,1].axhline(y=5, color='blue', ls='--', alpha=0.5, label='+5% target')
    axes[0,1].axhline(y=0, color='black', lw=1)
    axes[0,1].set_ylabel('Margin (WR - BE)'); axes[0,1].set_title('Profitability Margin', fontweight='bold')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=30)
    
    # 3: Signal frequency
    ann = [r["annual_signals"] for r in results.values()]
    axes[0,2].bar(labels_x, ann, color=['green' if a>=150 else 'orange' if a>=100 else 'red' for a in ann])
    axes[0,2].axhline(y=150, color='blue', ls='--', label='Min 150/yr')
    axes[0,2].set_ylabel('Annual Signals'); axes[0,2].set_title('Trade Frequency', fontweight='bold')
    axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)
    axes[0,2].tick_params(axis='x', rotation=30)
    
    # 4: Spread % of brick
    spr = [r["spread_pct_mean"] for r in results.values()]
    axes[1,0].plot(mults, spr, 'o-', color='crimson', lw=2)
    axes[1,0].axhline(y=5, color='green', ls='--', label='5% target')
    axes[1,0].set_xlabel('K'); axes[1,0].set_ylabel('Spread % of Brick')
    axes[1,0].set_title('Spread-to-Brick Ratio', fontweight='bold')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
    
    # 5: Brick duration
    dur = [r["median_dur_min"] for r in results.values()]
    axes[1,1].plot(mults, dur, 's-', color='steelblue', lw=2)
    axes[1,1].set_xlabel('K'); axes[1,1].set_ylabel('Median Duration (min)')
    axes[1,1].set_title('Brick Duration', fontweight='bold'); axes[1,1].grid(True, alpha=0.3)
    
    # 6: Composite score (margin × sqrt(frequency))
    scores = [max(0, r["margin"]) * np.sqrt(max(r["annual_signals"], 1)) for r in results.values()]
    axes[1,2].bar(labels_x, scores, color='steelblue')
    axes[1,2].set_ylabel('Score'); axes[1,2].set_title('Composite: Margin × √Frequency', fontweight='bold')
    axes[1,2].grid(True, alpha=0.3); axes[1,2].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    plt.savefig(RESULTS / "02_brick_multipliers.png", dpi=150)
    print(f"\n💾 Saved: 02_brick_multipliers.json, 02_brick_multipliers.png")

if __name__ == "__main__": main()
