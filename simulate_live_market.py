import random
import time

def simulate_market(num_trades=1000):
    print("--- Starting Synthetic Live Market Simulation ---")

    brick_size = 1.0
    slippage_threshold = brick_size * 0.08  # 8% slippage threshold from docs

    standard_wins = 0
    standard_losses = 0
    limit_order_fills = 0
    limit_order_losses = 0
    limit_order_misses = 0

    baiting_wins = 0
    baiting_losses = 0

    for _ in range(num_trades):
        # --- Simulate a Fast Market Breakout (Standard Signal) ---
        # Assuming the model correctly identifies a breakout roughly 70% of the time (pre-slippage)
        is_true_breakout = random.random() < 0.70

        # Simulate slippage during the breakout. Breakouts have higher slippage.
        actual_slippage = abs(random.gauss(0.05, 0.05)) # Mean 5% slippage, std 5%

        if actual_slippage > slippage_threshold:
            # Fallback to Limit Order triggered
            limit_order_fills += 1
            # In a true breakout, price never retraces to the limit order.
            if is_true_breakout:
                limit_order_misses += 1
            else:
                # If it's a false breakout, price retraces, hits the limit, and continues against us.
                limit_order_losses += 1
        else:
            # Market Order executed
            if is_true_breakout:
                # Spread eats into the TP. Need price to travel further.
                spread_spike = random.random() < 0.20 # 20% chance spread spikes and hits SL early
                if not spread_spike:
                    standard_wins += 1
                else:
                    standard_losses += 1
            else:
                standard_losses += 1

        # --- Simulate Baiting Strategy ---
        # Baiting occurs when model has low confidence. Often this is a ranging/choppy market.
        # In a ranging market, spread and noise frequently hit SLs in both directions.
        is_ranging = random.random() < 0.80 # Low model confidence implies ranging

        if is_ranging:
            # In a tight range, SL is hit easily due to spread
            if random.random() < 0.60: # 60% chance spread/noise hits SL
                baiting_losses += 1
            else:
                baiting_wins += 1
        else:
             # If it wasn't ranging, the model was just wrong about direction, baiting might work
             if random.random() < 0.50:
                 baiting_wins += 1
             else:
                 baiting_losses += 1


    print(f"\n--- Simulation Results ({num_trades} Signals) ---")
    print(f"Standard Trades (Market Orders):")
    market_total = standard_wins + standard_losses
    print(f"  Total: {market_total}")
    if market_total > 0:
         print(f"  Wins: {standard_wins} | Losses: {standard_losses} | Win Rate: {(standard_wins/market_total)*100:.1f}%")

    print(f"\nLimit Order Fallback (Adverse Selection):")
    print(f"  Total Attempted: {limit_order_fills}")
    print(f"  Missed (Price ran away): {limit_order_misses}")
    print(f"  Filled & Lost (False breakout hit limit): {limit_order_losses}")
    print(f"  Filled & Won: 0 (Assumed 0 in strong momentum regimes)")

    print(f"\nBaiting Strategy (Choppy Markets):")
    bait_total = baiting_wins + baiting_losses
    print(f"  Total: {bait_total}")
    print(f"  Wins: {baiting_wins} | Losses: {baiting_losses} | Win Rate: {(baiting_wins/bait_total)*100:.1f}%")

    print("\n--- Feasibility Conclusions ---")
    print("1. LIMIT ORDER ADVERSE SELECTION: The 8% slippage guard converts winning breakouts into misses, and losing false-breakouts into guaranteed losses.")
    print("2. BAITING STRATEGY: Trading inversely during low-confidence (ranging) regimes results in a negative edge due to spread/whipsaw.")
    print("3. SPREAD SENSITIVITY: 1:1 TP/SL ratios are easily destroyed by temporary spread widening, even if direction is correct.")

if __name__ == '__main__':
    simulate_market(5000)
