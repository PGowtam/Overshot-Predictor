"""
Test Suite: Full Bridge Integration (End-to-End)
==================================================
Simulates the complete EA ↔ Python communication flow using
a mock EA (Python TCP client) instead of real MT5.

Architecture:
  Python is SERVER on both ports (9000 tick, 9001 cmd).
  The EA (mock) connects as CLIENT to both.

Tests:
  1. Tick collection flow: DAYOPEN → HTICK batch → HDONE → live TICKs
  2. Order execution flow: Python sends BUY → EA confirms → state update
  3. Break-even MODIFYSL flow
  4. Full round-trip latency measurement
  5. Simultaneous tick streaming + command execution
"""

import pytest
import socket
import threading
import queue
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge.tick_receiver import TickReceiver
from bridge.command_sender import CommandSender


# ═══════════════════════════════════════════════════════════════════════
# Mock EA: Simulates TickSender.mq5 (dual-client architecture)
# ═══════════════════════════════════════════════════════════════════════

class MockEA:
    """
    Simulates the TickSender.mq5 EA for integration testing.

    Architecture matches the real EA — BOTH sockets are CLIENTS:
      - Connects to Python tick receiver on port 9000 (as TCP client)
      - Connects to Python command server on port 9001 (as TCP client)
      - Sends DAYOPEN, HTICK batch, HDONE, then live TICKs on tick socket
      - Reads commands from cmd socket, sends CONFIRM back on tick socket
    """

    def __init__(self, tick_port=29000, cmd_port=29001):
        self.tick_port = tick_port
        self.cmd_port = cmd_port
        self.tick_conn = None
        self.cmd_conn = None  # Client connection to Python's cmd server
        self._running = False

    def connect_tick(self):
        """Connect to Python tick receiver (as client)."""
        self.tick_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tick_conn.connect(('127.0.0.1', self.tick_port))

    def connect_cmd(self):
        """Connect to Python command server (as client)."""
        self.cmd_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cmd_conn.connect(('127.0.0.1', self.cmd_port))

    def send_tick_msg(self, msg):
        """Send a message on the tick channel."""
        if not msg.endswith('\n'):
            msg += '\n'
        self.tick_conn.sendall(msg.encode('utf-8'))

    def send_dayopen(self, price=2400.0):
        """Send DAYOPEN message."""
        self.send_tick_msg(f"DAYOPEN|{int(time.time()*1000)}|{price:.5f}")

    def send_history(self, count=100, base_price=2400.0):
        """Send HTICK batch + HDONE."""
        for i in range(count):
            bid = base_price + i * 0.01
            ask = bid + 0.10
            ts = int(time.time() * 1000) - (count - i) * 200
            self.send_tick_msg(
                f"HTICK|{ts}|{bid:.5f}|{ask:.5f}|1.00|1.00"
            )
        self.send_tick_msg(f"HDONE|{count}")

    def send_live_tick(self, bid=2400.0, ask=2400.10, bid_vol=1.0, ask_vol=1.0):
        """Send a single live TICK."""
        ts = int(time.time() * 1000)
        self.send_tick_msg(
            f"TICK|{ts}|{bid:.5f}|{ask:.5f}|{bid_vol:.2f}|{ask_vol:.2f}"
        )

    def send_confirm(self, req_id, ticket=12345678, status="OK"):
        """Send CONFIRM message back on tick socket."""
        self.send_tick_msg(f"CONFIRM|{req_id}|{ticket}|{status}")

    def read_command(self, timeout=2.0):
        """Read a command from the command socket (sent by Python)."""
        if self.cmd_conn is None:
            return None
        self.cmd_conn.settimeout(timeout)
        try:
            data = self.cmd_conn.recv(4096)
            return data.decode('utf-8').strip()
        except socket.timeout:
            return None

    def shutdown(self):
        """Close all connections."""
        for s in [self.tick_conn, self.cmd_conn]:
            if s:
                try:
                    s.close()
                except Exception:
                    pass


def setup_full_bridge(tick_port, cmd_port):
    """
    Helper: Start both Python servers, then connect the mock EA to both.
    Returns (receiver, sender, ea).
    
    Startup order:
      1. Start tick receiver (Python server on tick_port)
      2. Start command sender (Python server on cmd_port) in background
      3. Connect mock EA to tick_port (as client)
      4. Connect mock EA to cmd_port (as client) → triggers sender.accept()
    """
    receiver = TickReceiver(port=tick_port)
    receiver.start()
    time.sleep(0.1)

    sender = CommandSender(receiver.confirm_queue, port=cmd_port)

    # Start command server accept in background
    def accept_thread():
        sender.connect(timeout=5)

    t = threading.Thread(target=accept_thread, daemon=True)
    t.start()
    time.sleep(0.2)

    ea = MockEA(tick_port=tick_port, cmd_port=cmd_port)

    # Connect EA to tick receiver
    ea.connect_tick()
    time.sleep(0.1)
    ea.send_dayopen(2400.0)
    time.sleep(0.1)

    # Connect EA to command server (triggers sender.accept())
    ea.connect_cmd()
    t.join(timeout=3)
    time.sleep(0.1)

    return receiver, sender, ea


# ═══════════════════════════════════════════════════════════════════════
# Integration Test: Tick Collection Flow
# ═══════════════════════════════════════════════════════════════════════

class TestTickCollectionFlow:
    """Test the DAYOPEN → HTICK → HDONE → TICK flow."""

    def test_full_startup_sequence(self):
        """Complete startup: DAYOPEN → 100 HTICK → HDONE → 10 live TICKs."""
        receiver = TickReceiver(port=29100)
        receiver.start()
        time.sleep(0.3)

        ea = MockEA(tick_port=29100)
        try:
            ea.connect_tick()
            time.sleep(0.2)

            # Step 1: DAYOPEN
            ea.send_dayopen(2400.0)
            time.sleep(0.2)
            assert receiver.day_open_price == pytest.approx(2400.0, abs=1e-3)

            # Step 2: History batch
            ea.send_history(count=100, base_price=2399.0)
            time.sleep(0.5)
            assert len(receiver.history_ticks) == 100
            assert receiver.history_done.is_set()

            # Step 3: Live ticks
            for i in range(10):
                ea.send_live_tick(bid=2400.0 + i * 0.05, ask=2400.10 + i * 0.05)
                time.sleep(0.05)

            time.sleep(0.3)
            assert receiver.tick_queue.qsize() == 10
            assert receiver.tick_count == 10

        finally:
            ea.shutdown()
            receiver.stop()

    def test_zero_volume_ticks(self):
        """EA sends ticks with zero volume (volume fallback scenario)."""
        receiver = TickReceiver(port=29101)
        receiver.start()
        time.sleep(0.3)

        ea = MockEA(tick_port=29101)
        try:
            ea.connect_tick()
            time.sleep(0.2)

            ea.send_dayopen(2400.0)
            ea.send_live_tick(bid=2400.0, ask=2400.10, bid_vol=0.0, ask_vol=0.0)
            time.sleep(0.3)

            tick = receiver.tick_queue.get(timeout=1)
            assert tick['bid_vol'] == 0.0
            assert tick['ask_vol'] == 0.0

        finally:
            ea.shutdown()
            receiver.stop()


# ═══════════════════════════════════════════════════════════════════════
# Integration Test: Order Execution Flow
# ═══════════════════════════════════════════════════════════════════════

class TestOrderExecutionFlow:
    """Test the full BUY/SELL → CONFIRM round-trip."""

    def test_buy_order_round_trip(self):
        """
        Python sends BUY → EA receives → EA sends CONFIRM → Python gets result.
        """
        receiver, sender, ea = setup_full_bridge(29102, 29103)
        sender.CONFIRM_TIMEOUT = 3.0

        try:
            # Send BUY in background thread
            result_holder = [None]

            def do_buy():
                result_holder[0] = sender.buy(2400.10, 2393.02, 2407.18, 0.01)

            buy_thread = threading.Thread(target=do_buy)
            buy_thread.start()

            # EA reads the command from its cmd socket
            cmd = ea.read_command(timeout=2)
            assert cmd is not None
            parts = cmd.split('|')
            assert parts[0] == 'BUY'
            req_id = parts[-1]

            # EA sends CONFIRM back on tick socket
            ea.send_confirm(req_id, ticket=55443322, status="OK")

            # Wait for Python to receive CONFIRM
            buy_thread.join(timeout=5)

            assert result_holder[0] is not None
            assert result_holder[0]['ticket'] == 55443322
            assert result_holder[0]['status'] == 'OK'

        finally:
            sender.disconnect()
            ea.shutdown()
            receiver.stop()

    def test_modifysl_round_trip(self):
        """Python sends MODIFYSL → EA confirms → Python gets result."""
        receiver, sender, ea = setup_full_bridge(29104, 29105)
        sender.CONFIRM_TIMEOUT = 3.0

        try:
            result_holder = [None]

            def do_modify():
                result_holder[0] = sender.modify_sl(55443322, 2400.00)

            t = threading.Thread(target=do_modify)
            t.start()

            cmd = ea.read_command(timeout=2)
            assert cmd is not None
            parts = cmd.split('|')
            assert parts[0] == 'MODIFYSL'
            assert parts[1] == '55443322'
            assert parts[2] == '2400.00000'
            req_id = parts[-1]

            ea.send_confirm(req_id, ticket=55443322, status="OK")
            t.join(timeout=5)

            assert result_holder[0] is not None
            assert result_holder[0]['status'] == 'OK'

        finally:
            sender.disconnect()
            ea.shutdown()
            receiver.stop()

    def test_order_timeout_no_confirm(self):
        """BUY with no CONFIRM → returns None after timeout."""
        receiver, sender, ea = setup_full_bridge(29106, 29107)
        sender.CONFIRM_TIMEOUT = 1.5  # Short timeout for testing

        try:
            # Send BUY but EA will NOT send CONFIRM
            start = time.time()
            result = sender.buy(2400.0, 2393.0, 2407.0, 0.01)
            elapsed = time.time() - start

            assert result is None  # Timeout
            assert elapsed >= 1.0  # Should have waited ~1.5s
            assert elapsed < 5.0

        finally:
            sender.disconnect()
            ea.shutdown()
            receiver.stop()


# ═══════════════════════════════════════════════════════════════════════
# Integration Test: Simultaneous Tick + Command Flow
# ═══════════════════════════════════════════════════════════════════════

class TestSimultaneousFlow:
    """Test tick streaming and command execution happening concurrently."""

    def test_ticks_during_order(self):
        """
        Ticks continue streaming while a BUY order is being processed.
        """
        receiver, sender, ea = setup_full_bridge(29108, 29109)
        sender.CONFIRM_TIMEOUT = 3.0

        try:
            # Start streaming ticks in background
            def stream_ticks():
                for i in range(20):
                    ea.send_live_tick(bid=2400.0 + i * 0.01)
                    time.sleep(0.05)

            tick_thread = threading.Thread(target=stream_ticks, daemon=True)
            tick_thread.start()

            # Simultaneously send a BUY
            result_holder = [None]

            def do_buy():
                result_holder[0] = sender.buy(2400.10, 2393.02, 2407.18)

            buy_thread = threading.Thread(target=do_buy)
            buy_thread.start()

            # Give it a moment for command to arrive
            time.sleep(0.3)
            cmd = ea.read_command(timeout=2)
            if cmd:
                parts = cmd.split('|')
                req_id = parts[-1]
                ea.send_confirm(req_id, ticket=11223344)

            buy_thread.join(timeout=5)
            tick_thread.join(timeout=5)

            # Should have received both ticks and the order confirmation
            assert receiver.tick_count >= 10  # At least some ticks received
            assert result_holder[0] is not None
            assert result_holder[0]['ticket'] == 11223344

        finally:
            sender.disconnect()
            ea.shutdown()
            receiver.stop()


# ═══════════════════════════════════════════════════════════════════════
# Latency Measurement
# ═══════════════════════════════════════════════════════════════════════

class TestLatency:
    """Measure round-trip latency for the socket bridge."""

    def test_tick_receive_latency(self):
        """Measure time from sending a TICK to receiving it in Python."""
        receiver = TickReceiver(port=29110)
        receiver.start()
        time.sleep(0.3)

        ea = MockEA(tick_port=29110)
        try:
            ea.connect_tick()
            time.sleep(0.2)
            ea.send_dayopen(2400.0)

            # Warm up
            for _ in range(5):
                ea.send_live_tick()
                time.sleep(0.01)

            # Drain warmup ticks
            while not receiver.tick_queue.empty():
                receiver.tick_queue.get_nowait()

            # Measure
            latencies = []
            for i in range(50):
                t0 = time.perf_counter()
                ea.send_live_tick(bid=2400.0 + i * 0.01)
                try:
                    tick = receiver.tick_queue.get(timeout=1.0)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)  # ms
                except queue.Empty:
                    pass

            assert len(latencies) > 30  # At least 60% received

            avg_ms = sum(latencies) / len(latencies)
            p95_ms = sorted(latencies)[int(len(latencies) * 0.95)]

            print(f"\n  Tick receive latency: avg={avg_ms:.2f}ms  p95={p95_ms:.2f}ms")
            # Should be very fast on localhost
            assert p95_ms < 50, f"Tick receive p95 too high: {p95_ms:.2f}ms"

        finally:
            ea.shutdown()
            receiver.stop()


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
