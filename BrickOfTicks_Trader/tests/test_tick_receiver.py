"""
Test Suite: Tick Receiver (Phase 1.2 Verification)
====================================================
Tests for bridge/tick_receiver.py

Covers:
  - Protocol message parsing (TICK, HTICK, HDONE, DAYOPEN, CONFIRM, HEARTBEAT)
  - Queue overflow handling
  - Thread safety
  - Edge cases (malformed messages, empty lines)
"""

import pytest
import queue
import threading
import time
import socket

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge.tick_receiver import TickReceiver


# ═══════════════════════════════════════════════════════════════════════
# Basic Protocol Parsing Tests
# ═══════════════════════════════════════════════════════════════════════

class TestTickParsing:
    """Test TICK message parsing via feed_line()."""

    def test_tick_basic_parsing(self):
        """Feed a TICK line → assert all 5 fields are correct types."""
        r = TickReceiver()
        r.feed_line("TICK|1714900800123|2400.12000|2400.14000|3.50|2.10\n")

        assert r.tick_queue.qsize() == 1
        tick = r.tick_queue.get_nowait()

        assert tick['time_msc'] == 1714900800123
        assert isinstance(tick['time_msc'], int)
        assert tick['bid'] == pytest.approx(2400.12, abs=1e-5)
        assert isinstance(tick['bid'], float)
        assert tick['ask'] == pytest.approx(2400.14, abs=1e-5)
        assert isinstance(tick['ask'], float)
        assert tick['bid_vol'] == pytest.approx(3.50, abs=1e-2)
        assert isinstance(tick['bid_vol'], float)
        assert tick['ask_vol'] == pytest.approx(2.10, abs=1e-2)
        assert isinstance(tick['ask_vol'], float)

    def test_tick_zero_volume(self):
        """Tick with zero volume (triggers volume fallback in feature engine)."""
        r = TickReceiver()
        r.feed_line("TICK|1714900800123|2400.12000|2400.14000|0.00|0.00\n")

        tick = r.tick_queue.get_nowait()
        assert tick['bid_vol'] == 0.0
        assert tick['ask_vol'] == 0.0

    def test_tick_high_precision_prices(self):
        """5 decimal place prices are preserved."""
        r = TickReceiver()
        r.feed_line("TICK|1714900800123|2400.12345|2400.67891|1.00|1.00\n")

        tick = r.tick_queue.get_nowait()
        assert tick['bid'] == pytest.approx(2400.12345, abs=1e-6)
        assert tick['ask'] == pytest.approx(2400.67891, abs=1e-6)

    def test_tick_count_increments(self):
        """tick_count tracks total live ticks received."""
        r = TickReceiver()
        for i in range(50):
            r.feed_line(f"TICK|{1714900800000+i}|2400.00|2400.10|1.0|1.0\n")

        assert r.tick_count == 50
        assert r.tick_queue.qsize() == 50


class TestHistoryParsing:
    """Test HTICK and HDONE message parsing."""

    def test_htick_batch(self):
        """Feed 200 HTICK + HDONE|200 → assert correct counts."""
        r = TickReceiver()

        for i in range(200):
            r.feed_line(
                f"HTICK|{1714900800000+i}|{2400.0+i*0.01:.5f}|"
                f"{2400.1+i*0.01:.5f}|1.00|1.00\n"
            )
        r.feed_line("HDONE|200\n")

        assert len(r.history_ticks) == 200
        assert r.htick_count == 200
        assert r.history_done.is_set()

        # Verify first and last tick
        assert r.history_ticks[0]['time_msc'] == 1714900800000
        assert r.history_ticks[0]['bid'] == pytest.approx(2400.0, abs=1e-5)
        assert r.history_ticks[199]['time_msc'] == 1714900800199

    def test_hdone_event_set(self):
        """HDONE sets the history_done event."""
        r = TickReceiver()
        assert not r.history_done.is_set()

        r.feed_line("HDONE|0\n")
        assert r.history_done.is_set()

    def test_htick_not_in_tick_queue(self):
        """History ticks go to history_ticks list, NOT tick_queue."""
        r = TickReceiver()
        r.feed_line("HTICK|1714900800000|2400.00|2400.10|1.0|1.0\n")

        assert r.tick_queue.qsize() == 0  # Not in live queue
        assert len(r.history_ticks) == 1  # In history list


class TestDayOpenParsing:
    """Test DAYOPEN message parsing."""

    def test_dayopen_basic(self):
        """DAYOPEN sets day_open_price."""
        r = TickReceiver()
        r.feed_line("DAYOPEN|1714900800000|2398.50000\n")

        assert r.day_open_price == pytest.approx(2398.50, abs=1e-5)
        assert r.day_open_time == 1714900800000

    def test_dayopen_updates(self):
        """Multiple DAYOPEN messages update the price (rollover)."""
        r = TickReceiver()
        r.feed_line("DAYOPEN|1714900800000|2398.50000\n")
        assert r.day_open_price == pytest.approx(2398.50, abs=1e-5)

        r.feed_line("DAYOPEN|1714987200000|2415.00000\n")
        assert r.day_open_price == pytest.approx(2415.00, abs=1e-5)


class TestConfirmParsing:
    """Test CONFIRM message parsing."""

    def test_confirm_ok(self):
        """CONFIRM with OK status."""
        r = TickReceiver()
        r.feed_line("CONFIRM|abc12345|98765432|OK\n")

        assert r.confirm_queue.qsize() == 1
        conf = r.confirm_queue.get_nowait()
        assert conf['req_id'] == 'abc12345'
        assert conf['ticket'] == 98765432
        assert conf['status'] == 'OK'

    def test_confirm_error(self):
        """CONFIRM with ERROR status includes error code."""
        r = TickReceiver()
        r.feed_line("CONFIRM|def67890|0|ERROR|10006\n")

        conf = r.confirm_queue.get_nowait()
        assert conf['req_id'] == 'def67890'
        assert conf['ticket'] == 0
        assert conf['status'] == 'ERROR'
        assert conf['error_code'] == '10006'


class TestHeartbeat:
    """Test HEARTBEAT message handling."""

    def test_heartbeat_noop(self):
        """HEARTBEAT does not affect any queues or state."""
        r = TickReceiver()
        initial_tick_count = r.tick_count

        r.feed_line("HEARTBEAT|1714900800000\n")

        assert r.tick_queue.qsize() == 0
        assert r.tick_count == initial_tick_count
        assert len(r.history_ticks) == 0


# ═══════════════════════════════════════════════════════════════════════
# Queue Overflow Tests
# ═══════════════════════════════════════════════════════════════════════

class TestQueueOverflow:
    """Test tick queue overflow behavior."""

    def test_overflow_drops_oldest(self):
        """When queue is full, oldest tick is dropped."""
        r = TickReceiver()

        # Fill queue to capacity
        for i in range(10000):
            r.feed_line(f"TICK|{1714900800000+i}|2400.00|2400.10|1.0|1.0\n")

        assert r.tick_queue.qsize() == 10000

        # Push one more — should drop oldest
        r.feed_line("TICK|1714900810000|2500.00|2500.10|1.0|1.0\n")

        assert r.tick_queue.qsize() == 10000

        # The oldest tick (time_msc=1714900800000) should be gone
        # The newest tick (time_msc=1714900810000) should be present
        # Drain to find the newest
        found_newest = False
        while r.tick_queue.qsize() > 0:
            t = r.tick_queue.get_nowait()
            if t['time_msc'] == 1714900810000:
                found_newest = True
        assert found_newest


# ═══════════════════════════════════════════════════════════════════════
# Thread Safety Tests
# ═══════════════════════════════════════════════════════════════════════

class TestThreadSafety:
    """Test concurrent access to tick queue."""

    def test_concurrent_producers(self):
        """Two threads pushing 500 ticks each → no crash, correct count."""
        r = TickReceiver()

        def produce(offset):
            for i in range(500):
                r.feed_line(
                    f"TICK|{1714900800000+offset+i}|2400.00|2400.10|1.0|1.0\n")

        t1 = threading.Thread(target=produce, args=(0,))
        t2 = threading.Thread(target=produce, args=(100000,))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Should have exactly 1000 ticks (queue max is 10000 so no overflow)
        assert r.tick_count == 1000
        assert r.tick_queue.qsize() == 1000


# ═══════════════════════════════════════════════════════════════════════
# Edge Cases & Malformed Messages
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test error handling for malformed or unexpected input."""

    def test_empty_line_ignored(self):
        """Empty lines are silently ignored."""
        r = TickReceiver()
        r.feed_line("\n")
        r.feed_line("   \n")
        r.feed_line("")

        assert r.tick_queue.qsize() == 0
        assert r.tick_count == 0

    def test_unknown_message_type(self):
        """Unknown message types are logged but don't crash."""
        r = TickReceiver()
        r.feed_line("FOOBAR|123|456\n")

        assert r.tick_queue.qsize() == 0  # Not added to queue

    def test_malformed_tick_missing_fields(self):
        """TICK with too few fields doesn't crash."""
        r = TickReceiver()
        r.feed_line("TICK|1714900800123|2400.12\n")  # Missing 3 fields

        assert r.tick_queue.qsize() == 0  # Parsing failed gracefully
        assert r.tick_count == 0

    def test_malformed_tick_bad_float(self):
        """TICK with non-numeric price doesn't crash."""
        r = TickReceiver()
        r.feed_line("TICK|1714900800123|INVALID|2400.14|1.0|1.0\n")

        assert r.tick_queue.qsize() == 0  # Parsing failed

    def test_multiple_messages_in_one_chunk(self):
        """Multiple newline-separated messages in a single feed."""
        r = TickReceiver()

        # Simulate receiving a chunk with 3 messages
        lines = (
            "DAYOPEN|1714900800000|2400.00\n"
            "TICK|1714900800001|2400.10|2400.20|1.0|1.0\n"
            "TICK|1714900800002|2400.15|2400.25|2.0|2.0\n"
        )
        for line in lines.strip().split('\n'):
            r.feed_line(line)

        assert r.day_open_price == pytest.approx(2400.00, abs=1e-5)
        assert r.tick_queue.qsize() == 2
        assert r.tick_count == 2


# ═══════════════════════════════════════════════════════════════════════
# Socket Integration Tests (localhost loopback)
# ═══════════════════════════════════════════════════════════════════════

class TestSocketIntegration:
    """Test actual TCP socket communication on localhost."""

    def test_socket_receive_tick(self):
        """Start receiver, connect a mock client, send a TICK → verify."""
        r = TickReceiver(port=19000)  # Use high port to avoid conflicts
        r.start()
        time.sleep(0.3)

        # Connect as a mock EA
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(('127.0.0.1', 19000))
            time.sleep(0.2)

            # Send DAYOPEN + TICK
            client.sendall(b"DAYOPEN|1714900800000|2400.00000\n")
            client.sendall(b"TICK|1714900800001|2400.10000|2400.20000|1.00|1.00\n")
            time.sleep(0.3)

            assert r.day_open_price == pytest.approx(2400.0, abs=1e-5)
            assert r.tick_queue.qsize() >= 1

            tick = r.tick_queue.get_nowait()
            assert tick['bid'] == pytest.approx(2400.1, abs=1e-5)
        finally:
            client.close()
            r.stop()

    def test_socket_receive_history_batch(self):
        """Send HTICK batch + HDONE via socket → verify history."""
        r = TickReceiver(port=19001)
        r.start()
        time.sleep(0.3)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(('127.0.0.1', 19001))
            time.sleep(0.2)

            # Send history batch
            for i in range(50):
                msg = f"HTICK|{1714900800000+i}|{2400.0+i*0.01:.5f}|{2400.1+i*0.01:.5f}|1.00|1.00\n"
                client.sendall(msg.encode('utf-8'))

            client.sendall(b"HDONE|50\n")
            time.sleep(0.5)

            assert len(r.history_ticks) == 50
            assert r.history_done.is_set()
        finally:
            client.close()
            r.stop()

    def test_socket_confirm_received(self):
        """Send CONFIRM via socket → verify confirm_queue."""
        r = TickReceiver(port=19002)
        r.start()
        time.sleep(0.3)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client.connect(('127.0.0.1', 19002))
            time.sleep(0.2)

            client.sendall(b"CONFIRM|req001|12345678|OK\n")
            time.sleep(0.3)

            assert r.confirm_queue.qsize() >= 1
            conf = r.confirm_queue.get_nowait()
            assert conf['req_id'] == 'req001'
            assert conf['ticket'] == 12345678
            assert conf['status'] == 'OK'
        finally:
            client.close()
            r.stop()


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
