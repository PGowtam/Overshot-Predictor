"""
Tick Collector — TCP Server for MQL5 TickExporter EA
====================================================
Receives historical tick data from MT5 via socket and saves to parquet.
Run this FIRST, then attach TickExporter EA to a chart in MT5.

Usage:
    python src/tick_collector.py [--port 9100] [--output data/xauusd_ticks_2026.parquet]
"""

import socket
import argparse
import time
import sys
import os
from pathlib import Path

# Add project root to path for potential future imports
BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Tick Collector for TickExporter EA")
    parser.add_argument("--port", type=int, default=9100, help="TCP port to listen on")
    parser.add_argument("--output", type=str, 
                        default=str(BASE_DIR / "data" / "xauusd_ticks_2026.parquet"),
                        help="Output parquet file path")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=" * 60)
    print(" Tick Collector — Waiting for TickExporter EA")
    print(f" Listening on 127.0.0.1:{args.port}")
    print(f" Output: {args.output}")
    print("=" * 60)
    
    # Create TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', args.port))
    server.listen(1)
    
    print("⏳ Waiting for EA connection...")
    conn, addr = server.accept()
    print(f"✓ EA connected from {addr}")
    
    # Receive ticks
    ticks = []
    buf = ''
    total_received = 0
    last_report = time.time()
    done = False
    
    try:
        while not done:
            data = conn.recv(65536)
            if not data:
                print("⚠️  Connection closed by EA before HDONE.")
                break
                
            buf += data.decode('utf-8', errors='replace')
            
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('|')
                msg_type = parts[0]
                
                if msg_type == 'HTICK':
                    try:
                        tick = {
                            'time_msc': int(parts[1]),
                            'bid': float(parts[2]),
                            'ask': float(parts[3]),
                            'bid_vol': float(parts[4]),
                            'ask_vol': float(parts[5])
                        }
                        ticks.append(tick)
                        total_received += 1
                        
                        # Progress report every 5 seconds
                        if time.time() - last_report > 5.0:
                            print(f"  📊 Received {total_received:,} ticks...")
                            last_report = time.time()
                            
                    except (IndexError, ValueError) as e:
                        print(f"  ⚠️  Parse error: {line[:80]} — {e}")
                        
                elif msg_type == 'HDONE':
                    ea_count = int(parts[1])
                    print(f"\n✓ HDONE received. EA sent: {ea_count}, Python received: {total_received}")
                    done = True
                    
                else:
                    print(f"  Unknown message: {msg_type}")
                    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted. Saving what we have...")
    finally:
        conn.close()
        server.close()
    
    if total_received == 0:
        print("❌ No ticks received. Nothing to save.")
        return
        
    # Convert to parquet
    print(f"\n💾 Converting {total_received:,} ticks to parquet...")
    
    try:
        import pandas as pd
        df = pd.DataFrame(ticks)
        
        # Sort by time (should already be sorted but safety first)
        df = df.sort_values('time_msc').reset_index(drop=True)
        
        # Deduplicate consecutive identical timestamps
        before = len(df)
        df = df.drop_duplicates(subset=['time_msc'], keep='first')
        after = len(df)
        if before != after:
            print(f"  Deduplication: {before:,} → {after:,} ({before - after:,} duplicates removed)")
        
        df.to_parquet(args.output, index=False)
        
        # Print summary
        from datetime import datetime, timezone
        first_ts = datetime.fromtimestamp(df['time_msc'].iloc[0] / 1000, tz=timezone.utc)
        last_ts = datetime.fromtimestamp(df['time_msc'].iloc[-1] / 1000, tz=timezone.utc)
        
        print(f"\n{'=' * 60}")
        print(f" COLLECTION COMPLETE")
        print(f" Total ticks: {len(df):,}")
        print(f" Date range:  {first_ts.strftime('%Y-%m-%d %H:%M')} → {last_ts.strftime('%Y-%m-%d %H:%M')} UTC")
        print(f" Price range: {df['bid'].min():.2f} → {df['bid'].max():.2f}")
        print(f" File size:   {os.path.getsize(args.output) / 1e6:.1f} MB")
        print(f" Saved to:    {args.output}")
        print(f"{'=' * 60}")
        
    except ImportError:
        print("ERROR: pandas not available. Saving as CSV fallback.")
        import csv
        csv_path = args.output.replace('.parquet', '.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time_msc', 'bid', 'ask', 'bid_vol', 'ask_vol'])
            writer.writeheader()
            writer.writerows(ticks)
        print(f"Saved {total_received:,} ticks to {csv_path}")


if __name__ == "__main__":
    main()
