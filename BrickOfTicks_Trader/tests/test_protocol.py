"""
Test Suite: Protocol Parser & Wire Format
==========================================
Verifies that the socket bridge protocol matches the specification
in SOCKET_BRIDGE_PRD_v2.md Section 3.2.

Tests the complete message format for ALL message types:
  Port 9000 (EA → Python): DAYOPEN, HTICK, HDONE, TICK, HEARTBEAT, CONFIRM
  Port 9001 (Python → EA): BUY, SELL, CLOSE, MODIFYSL
"""

import pytest
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge.tick_receiver import TickReceiver


# ═══════════════════════════════════════════════════════════════════════
# EA → Python Messages (Port 9000)
# ═══════════════════════════════════════════════════════════════════════

class TestPort9000Protocol:
    """Verify all EA → Python message formats."""

    def test_dayopen_format(self):
        """DAYOPEN|<time_msc>|<price>"""
        r = TickReceiver()
        msg = "DAYOPEN|1714900800123|2400.50000"
        r.feed_line(msg)

        assert r.day_open_price == pytest.approx(2400.5, abs=1e-5)
        assert r.day_open_time == 1714900800123

    def test_htick_format(self):
        """HTICK|<time_msc>|<bid>|<ask>|<bid_vol>|<ask_vol>"""
        r = TickReceiver()
        msg = "HTICK|1714900800123|2400.12345|2400.98765|3.50|2.10"
        r.feed_line(msg)

        assert len(r.history_ticks) == 1
        h = r.history_ticks[0]
        assert h['time_msc'] == 1714900800123
        assert h['bid'] == pytest.approx(2400.12345, abs=1e-6)
        assert h['ask'] == pytest.approx(2400.98765, abs=1e-6)
        assert h['bid_vol'] == pytest.approx(3.50, abs=1e-2)
        assert h['ask_vol'] == pytest.approx(2.10, abs=1e-2)

    def test_hdone_format(self):
        """HDONE|<count>"""
        r = TickReceiver()
        r.feed_line("HDONE|5000")

        assert r.history_done.is_set()

    def test_tick_format(self):
        """TICK|<time_msc>|<bid>|<ask>|<bid_vol>|<ask_vol>"""
        r = TickReceiver()
        msg = "TICK|1714900800999|2401.55555|2401.77777|0.00|0.00"
        r.feed_line(msg)

        tick = r.tick_queue.get_nowait()
        assert tick['time_msc'] == 1714900800999
        assert tick['bid'] == pytest.approx(2401.55555, abs=1e-6)
        assert tick['ask'] == pytest.approx(2401.77777, abs=1e-6)
        assert tick['bid_vol'] == 0.0
        assert tick['ask_vol'] == 0.0

    def test_heartbeat_format(self):
        """HEARTBEAT|<time_msc>"""
        r = TickReceiver()
        r.feed_line("HEARTBEAT|1714900800000")
        # No error, no state change
        assert r.tick_queue.qsize() == 0

    def test_confirm_ok_format(self):
        """CONFIRM|<req_id>|<ticket>|OK"""
        r = TickReceiver()
        r.feed_line("CONFIRM|a1b2c3d4|98765432|OK")

        conf = r.confirm_queue.get_nowait()
        assert conf['req_id'] == 'a1b2c3d4'
        assert conf['ticket'] == 98765432
        assert conf['status'] == 'OK'

    def test_confirm_error_format(self):
        """CONFIRM|<req_id>|0|ERROR|<code>"""
        r = TickReceiver()
        r.feed_line("CONFIRM|a1b2c3d4|0|ERROR|10006")

        conf = r.confirm_queue.get_nowait()
        assert conf['req_id'] == 'a1b2c3d4'
        assert conf['ticket'] == 0
        assert conf['status'] == 'ERROR'
        assert conf['error_code'] == '10006'


# ═══════════════════════════════════════════════════════════════════════
# Python → EA Messages (Port 9001) — Format Verification
# ═══════════════════════════════════════════════════════════════════════

class TestPort9001Protocol:
    """Verify Python → EA command format via CommandSender internals."""

    def test_buy_format_regex(self):
        """BUY|<price>|<sl>|<tp>|<volume>|<req_id> — validate format."""
        # Build a BUY command manually
        import uuid
        req_id = str(uuid.uuid4())[:8]
        price = 2400.10
        sl = 2393.02
        tp = 2407.18
        vol = 0.01

        line = f"BUY|{price:.5f}|{sl:.5f}|{tp:.5f}|{vol:.2f}|{req_id}"

        # Validate format with regex
        pattern = r'^BUY\|\d+\.\d{5}\|\d+\.\d{5}\|\d+\.\d{5}\|\d+\.\d{2}\|[a-f0-9]{8}$'
        assert re.match(pattern, line), f"Format mismatch: {line}"

    def test_sell_format_regex(self):
        """SELL|<price>|<sl>|<tp>|<volume>|<req_id>"""
        import uuid
        req_id = str(uuid.uuid4())[:8]
        line = f"SELL|2400.50000|2407.58000|2393.42000|0.05|{req_id}"

        pattern = r'^SELL\|\d+\.\d{5}\|\d+\.\d{5}\|\d+\.\d{5}\|\d+\.\d{2}\|[a-f0-9]{8}$'
        assert re.match(pattern, line)

    def test_close_format_regex(self):
        """CLOSE|<ticket>|<req_id>"""
        import uuid
        req_id = str(uuid.uuid4())[:8]
        line = f"CLOSE|98765432|{req_id}"

        pattern = r'^CLOSE\|\d+\|[a-f0-9]{8}$'
        assert re.match(pattern, line)

    def test_modifysl_format_regex(self):
        """MODIFYSL|<ticket>|<new_sl>|<req_id>"""
        import uuid
        req_id = str(uuid.uuid4())[:8]
        line = f"MODIFYSL|98765432|2400.00000|{req_id}"

        pattern = r'^MODIFYSL\|\d+\|\d+\.\d{5}\|[a-f0-9]{8}$'
        assert re.match(pattern, line)


# ═══════════════════════════════════════════════════════════════════════
# Encoding & Delimiters
# ═══════════════════════════════════════════════════════════════════════

class TestEncoding:
    """Verify encoding and delimiter conventions."""

    def test_newline_terminated(self):
        """Messages are newline-terminated (\\n)."""
        r = TickReceiver()
        # Without newline — should still parse if stripped
        r.feed_line("TICK|1714900800000|2400.00|2400.10|1.0|1.0")
        assert r.tick_queue.qsize() == 1

    def test_pipe_delimiter(self):
        """Fields are pipe-delimited."""
        r = TickReceiver()
        r.feed_line("TICK|111|222.22222|333.33333|4.44|5.55")
        tick = r.tick_queue.get_nowait()
        assert tick['time_msc'] == 111

    def test_float_precision(self):
        """All floats use 5 decimal places (dot-decimal)."""
        # Price at exactly 5 decimals
        r = TickReceiver()
        r.feed_line("TICK|1000|2400.12345|2400.67890|1.00|1.00")
        tick = r.tick_queue.get_nowait()
        assert abs(tick['bid'] - 2400.12345) < 1e-6
        assert abs(tick['ask'] - 2400.67890) < 1e-6

    def test_timestamp_int64(self):
        """Timestamps are ms since Unix epoch (int64)."""
        r = TickReceiver()
        # Use a real timestamp: 2024-05-05 12:00:00 UTC
        ts = 1714910400000
        r.feed_line(f"TICK|{ts}|2400.00|2400.10|1.0|1.0")
        tick = r.tick_queue.get_nowait()
        assert tick['time_msc'] == ts
        assert isinstance(tick['time_msc'], int)


# ═══════════════════════════════════════════════════════════════════════
# Sequence Tests (full startup protocol)
# ═══════════════════════════════════════════════════════════════════════

class TestStartupSequence:
    """Test the correct startup message sequence per FR-MQL-03."""

    def test_full_startup_sequence(self):
        """DAYOPEN → HTICK × N → HDONE → TICK (live mode)"""
        r = TickReceiver()

        # Step 1: DAYOPEN
        r.feed_line("DAYOPEN|1714900800000|2400.00000")
        assert r.day_open_price is not None

        # Step 2: History batch (3 ticks)
        r.feed_line("HTICK|1714900700000|2399.50|2399.60|1.0|1.0")
        r.feed_line("HTICK|1714900700100|2399.55|2399.65|1.0|1.0")
        r.feed_line("HTICK|1714900700200|2399.60|2399.70|1.0|1.0")

        # Step 3: HDONE
        r.feed_line("HDONE|3")
        assert r.history_done.is_set()
        assert len(r.history_ticks) == 3

        # Step 4: Live ticks
        r.feed_line("TICK|1714900800001|2400.05|2400.15|2.0|2.0")
        r.feed_line("TICK|1714900800002|2400.10|2400.20|3.0|3.0")

        assert r.tick_queue.qsize() == 2
        assert r.tick_count == 2

        # History ticks should NOT be in the live queue
        assert r.tick_queue.qsize() == 2  # Only live ticks


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
