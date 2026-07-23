"""
BrickOfTicks Socket Bridge — Live Command Audit (Phase 0.3 Verification)
=========================================================================
Tests the full BUY/SELL/MODIFYSL/CLOSE command round-trip against the
live MT5 EA on a DEMO account.

This script:
  1. Starts TickReceiver on port 9000 (to receive CONFIRMs)
  2. Waits for EA to connect and send DAYOPEN
  3. Connects CommandSender to EA on port 9001
  4. Sends test BUY → verifies CONFIRM|OK
  5. Sends MODIFYSL (break-even) → verifies CONFIRM|OK
  6. Sends CLOSE → verifies CONFIRM|OK
  7. Sends test SELL → verifies CONFIRM|OK
  8. Closes SELL → verifies CONFIRM|OK
  9. Generates command audit report

Requirements:
  - MT5 running with TickSender.mq5 attached to XAUUSD chart (demo account)
  - Demo account funded (even paper money)
  - EA must have both tick_socket AND cmd_socket connected

Usage:
  python -m tests.test_live_commands
  python -m tests.test_live_commands --volume=0.01 --tick-port=9000 --cmd-port=9001

WARNING: This WILL execute real orders on your connected account.
         Only run on a DEMO account with minimum lot size.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge.tick_receiver import TickReceiver
from bridge.command_sender import CommandSender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('cmd_audit')

# ─── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "bridge" / "logs" / "audit"


def print_header():
    print(f"\n{'='*60}")
    print(f"  COMMAND AUDIT — Phase 0.3 Verification")
    print(f"  ⚠️  This will execute REAL orders on your MT5 account!")
    print(f"  Make sure you are on a DEMO account with minimum lots.")
    print(f"{'='*60}\n")


def wait_for_ea(receiver, timeout=30):
    """Wait for EA to connect and send DAYOPEN."""
    print("Waiting for EA to connect and send DAYOPEN...")
    start = time.time()
    while receiver.day_open_price is None:
        if time.time() - start > timeout:
            print("ERROR: No DAYOPEN received. Is the EA running and connected?")
            return False
        time.sleep(0.5)

    print(f"  ✓ EA connected. Day open price: {receiver.day_open_price}")

    # Wait for history (optional for command testing)
    print("Waiting for history batch (5s max)...")
    receiver.history_done.wait(timeout=5)
    if receiver.history_done.is_set():
        print(f"  ✓ History received: {receiver.htick_count} ticks")
    else:
        print("  ⚠️ No history received (OK for command testing)")

    return True


def get_current_prices(receiver, timeout=5):
    """Get the latest bid/ask from the tick stream."""
    deadline = time.time() + timeout
    last_tick = None
    while time.time() < deadline:
        try:
            tick = receiver.tick_queue.get(timeout=0.5)
            last_tick = tick
        except Exception:
            if last_tick:
                break
            continue
    return last_tick


def run_command_audit(tick_port=9000, cmd_port=9001, volume=0.01):
    """Execute the full command audit sequence."""
    print_header()

    results = {
        'timestamp': datetime.now().isoformat(),
        'tick_port': tick_port,
        'cmd_port': cmd_port,
        'volume': volume,
        'tests': {}
    }

    # ─── Step 1: Start BOTH servers before EA connects ────────
    # The EA connects to BOTH ports during OnInit() or reconnect.
    # Both servers must be listening before the EA tries to connect.
    print("Step 1: Starting TickReceiver (port", tick_port, ") + CommandSender (port", cmd_port, ")")

    receiver = TickReceiver(port=tick_port)
    receiver.start()
    time.sleep(0.1)

    sender = CommandSender(receiver.confirm_queue, port=cmd_port)

    # Start command server's accept() in a background thread
    # (it blocks until EA connects)
    import threading
    cmd_connect_result = [None]

    def accept_cmd():
        try:
            sender.connect(timeout=60)
            cmd_connect_result[0] = True
        except Exception as e:
            cmd_connect_result[0] = str(e)

    cmd_thread = threading.Thread(target=accept_cmd, daemon=True)
    cmd_thread.start()
    time.sleep(0.3)

    print("  Both servers listening. Waiting for EA to connect (up to 60s)...")
    print("  ⚠️  You may need to re-attach the EA to the chart to trigger reconnect.")

    # ─── Step 2: Wait for EA to connect on tick channel ──────
    if not wait_for_ea(receiver, timeout=60):
        receiver.stop()
        sender.disconnect()
        results['verdict'] = 'FAILED - EA not connected'
        return results

    # ─── Step 3: Wait for EA to connect on command channel ───
    print("\nStep 3: Waiting for command channel connection...")
    cmd_thread.join(timeout=10)

    if cmd_connect_result[0] is True:
        print("  ✓ CommandSender: EA connected on command channel")
        results['tests']['cmd_connect'] = {'PASS': True}
    else:
        print(f"  ❌ Command channel failed: {cmd_connect_result[0]}")
        results['tests']['cmd_connect'] = {'PASS': False, 'error': str(cmd_connect_result[0])}
        results['verdict'] = 'FAILED - Command socket not available'
        receiver.stop()
        sender.disconnect()
        return results

    # ─── Step 4: Get current prices ─────────────────────────────
    print("\nStep 4: Getting current market prices...")
    tick = get_current_prices(receiver, timeout=5)
    if tick is None:
        print("  ❌ No ticks received. Market may be closed.")
        results['verdict'] = 'FAILED - No market data'
        sender.disconnect()
        receiver.stop()
        return results

    bid = tick['bid']
    ask = tick['ask']
    spread = ask - bid
    print(f"  ✓ Current: bid={bid:.5f}  ask={ask:.5f}  spread={spread:.5f}")
    results['market'] = {'bid': bid, 'ask': ask, 'spread': round(spread, 5)}

    # Compute SL/TP for test trades
    # Use brick_size derived from day_open * K
    K = 0.00295
    day_open = receiver.day_open_price
    brick_size = day_open * K
    sl_dist = brick_size
    tp_dist = brick_size

    print(f"  brick_size = {day_open} × {K} = {brick_size:.5f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: BUY Order
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print(f"  TEST 1: BUY Order")
    print(f"{'─'*50}")

    buy_price = ask  # Market buy at ask
    buy_sl = buy_price - sl_dist
    buy_tp = buy_price + tp_dist

    print(f"  Sending BUY: price={buy_price:.5f}  sl={buy_sl:.5f}  tp={buy_tp:.5f}  vol={volume}")

    t0 = time.time()
    buy_result = sender.buy(buy_price, buy_sl, buy_tp, volume)
    buy_latency = (time.time() - t0) * 1000

    if buy_result and buy_result['status'] == 'OK':
        buy_ticket = buy_result['ticket']
        print(f"  ✅ BUY CONFIRMED: ticket={buy_ticket}  latency={buy_latency:.0f}ms")
        results['tests']['buy'] = {
            'PASS': True,
            'ticket': buy_ticket,
            'latency_ms': round(buy_latency, 1),
            'price': buy_price,
            'sl': buy_sl,
            'tp': buy_tp
        }
    elif buy_result and 'ERROR' in buy_result['status']:
        print(f"  ❌ BUY ERROR: {buy_result}")
        results['tests']['buy'] = {'PASS': False, 'error': str(buy_result)}
        buy_ticket = None
    else:
        print(f"  ❌ BUY TIMEOUT (no CONFIRM in 5s)")
        results['tests']['buy'] = {'PASS': False, 'error': 'TIMEOUT'}
        buy_ticket = None

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: MODIFYSL (break-even trigger simulation)
    # ═══════════════════════════════════════════════════════════════
    if buy_ticket:
        time.sleep(1)  # Wait a moment before modifying
        print(f"\n{'─'*50}")
        print(f"  TEST 2: MODIFYSL (break-even)")
        print(f"{'─'*50}")

        new_sl = buy_price  # Move SL to entry (break-even)
        print(f"  Sending MODIFYSL: ticket={buy_ticket}  new_sl={new_sl:.5f}")

        t0 = time.time()
        modify_result = sender.modify_sl(buy_ticket, new_sl)
        modify_latency = (time.time() - t0) * 1000

        if modify_result and modify_result['status'] == 'OK':
            print(f"  ✅ MODIFYSL CONFIRMED: latency={modify_latency:.0f}ms")
            results['tests']['modifysl'] = {
                'PASS': True,
                'latency_ms': round(modify_latency, 1),
                'new_sl': new_sl
            }
        else:
            print(f"  ❌ MODIFYSL FAILED: {modify_result}")
            results['tests']['modifysl'] = {
                'PASS': False,
                'error': str(modify_result)
            }

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: CLOSE the BUY position
    # ═══════════════════════════════════════════════════════════════
    if buy_ticket:
        time.sleep(1)
        print(f"\n{'─'*50}")
        print(f"  TEST 3: CLOSE BUY position")
        print(f"{'─'*50}")

        print(f"  Sending CLOSE: ticket={buy_ticket}")

        t0 = time.time()
        close_result = sender.close_position(buy_ticket)
        close_latency = (time.time() - t0) * 1000

        if close_result and close_result['status'] == 'OK':
            print(f"  ✅ CLOSE CONFIRMED: latency={close_latency:.0f}ms")
            results['tests']['close_buy'] = {
                'PASS': True,
                'latency_ms': round(close_latency, 1)
            }
        else:
            print(f"  ❌ CLOSE FAILED: {close_result}")
            results['tests']['close_buy'] = {
                'PASS': False,
                'error': str(close_result)
            }

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: SELL Order
    # ═══════════════════════════════════════════════════════════════
    time.sleep(2)
    # Refresh prices
    tick2 = get_current_prices(receiver, timeout=3) or tick
    bid = tick2['bid']
    ask = tick2['ask']

    print(f"\n{'─'*50}")
    print(f"  TEST 4: SELL Order")
    print(f"{'─'*50}")

    sell_price = bid  # Market sell at bid
    sell_sl = sell_price + sl_dist
    sell_tp = sell_price - tp_dist

    print(f"  Sending SELL: price={sell_price:.5f}  sl={sell_sl:.5f}  tp={sell_tp:.5f}  vol={volume}")

    t0 = time.time()
    sell_result = sender.sell(sell_price, sell_sl, sell_tp, volume)
    sell_latency = (time.time() - t0) * 1000

    if sell_result and sell_result['status'] == 'OK':
        sell_ticket = sell_result['ticket']
        print(f"  ✅ SELL CONFIRMED: ticket={sell_ticket}  latency={sell_latency:.0f}ms")
        results['tests']['sell'] = {
            'PASS': True,
            'ticket': sell_ticket,
            'latency_ms': round(sell_latency, 1)
        }
    elif sell_result and 'ERROR' in sell_result['status']:
        print(f"  ❌ SELL ERROR: {sell_result}")
        results['tests']['sell'] = {'PASS': False, 'error': str(sell_result)}
        sell_ticket = None
    else:
        print(f"  ❌ SELL TIMEOUT")
        results['tests']['sell'] = {'PASS': False, 'error': 'TIMEOUT'}
        sell_ticket = None

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: CLOSE the SELL position
    # ═══════════════════════════════════════════════════════════════
    if sell_ticket:
        time.sleep(1)
        print(f"\n{'─'*50}")
        print(f"  TEST 5: CLOSE SELL position")
        print(f"{'─'*50}")

        print(f"  Sending CLOSE: ticket={sell_ticket}")

        t0 = time.time()
        close_sell_result = sender.close_position(sell_ticket)
        close_sell_latency = (time.time() - t0) * 1000

        if close_sell_result and close_sell_result['status'] == 'OK':
            print(f"  ✅ CLOSE CONFIRMED: latency={close_sell_latency:.0f}ms")
            results['tests']['close_sell'] = {
                'PASS': True,
                'latency_ms': round(close_sell_latency, 1)
            }
        else:
            print(f"  ❌ CLOSE FAILED: {close_sell_result}")
            results['tests']['close_sell'] = {
                'PASS': False,
                'error': str(close_sell_result)
            }

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    sender.disconnect()
    receiver.stop()

    total_tests = len(results['tests'])
    passed = sum(1 for t in results['tests'].values() if t.get('PASS'))
    failed = total_tests - passed

    latencies = [t['latency_ms'] for t in results['tests'].values()
                 if 'latency_ms' in t]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0

    results['summary'] = {
        'total_tests': total_tests,
        'passed': passed,
        'failed': failed,
        'avg_latency_ms': round(avg_latency, 1),
        'max_latency_ms': round(max_latency, 1)
    }

    all_passed = failed == 0
    results['verdict'] = 'PASS' if all_passed else 'FAIL'

    print(f"\n{'='*60}")
    print(f"  COMMAND AUDIT — RESULTS")
    print(f"{'='*60}")
    print(f"\n  Tests: {passed}/{total_tests} passed")
    for name, test in results['tests'].items():
        status = '✅' if test.get('PASS') else '❌'
        latency = f"  ({test['latency_ms']:.0f}ms)" if 'latency_ms' in test else ''
        print(f"    {status} {name}{latency}")

    if latencies:
        print(f"\n  Avg latency: {avg_latency:.0f}ms")
        print(f"  Max latency: {max_latency:.0f}ms")

    print(f"\n  {'='*56}")
    print(f"  VERDICT: {'✅ ALL COMMANDS OPERATIONAL' if all_passed else '❌ COMMAND FAILURES DETECTED'}")
    print(f"  {'='*56}\n")

    # Save results
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = AUDIT_DIR / f"command_audit_{date_str}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ Results saved to {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="BrickOfTicks Live Command Audit (Phase 0.3)")
    parser.add_argument('--volume', type=float, default=0.01,
                        help='Lot size for test trades (default: 0.01)')
    parser.add_argument('--tick-port', type=int, default=9000,
                        help='Tick channel port (default: 9000)')
    parser.add_argument('--cmd-port', type=int, default=9001,
                        help='Command channel port (default: 9001)')

    args = parser.parse_args()
    run_command_audit(
        tick_port=args.tick_port,
        cmd_port=args.cmd_port,
        volume=args.volume
    )


if __name__ == '__main__':
    main()
