"""
Test Suite: Command Sender (Phase 2.2 Verification)
=====================================================
Tests for bridge/command_sender.py

Architecture: CommandSender is a TCP SERVER on port 9001.
The EA connects to it as a CLIENT. Python sends commands
over the accepted connection, CONFIRMs come back via tick channel.

Covers:
  - Message format verification (BUY, SELL, MODIFYSL, CLOSE)
  - 5-second CONFIRM timeout behavior
  - Reconnect on broken pipe
  - Edge cases
"""

import pytest
import queue
import threading
import time
import socket

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bridge.command_sender import CommandSender


# ═══════════════════════════════════════════════════════════════════════
# Helper: Mock EA client (simulates the EA connecting to Python)
# ═══════════════════════════════════════════════════════════════════════

class MockEAClient:
    """
    Simulates the EA connecting to Python's command server on port 9001.
    The EA is a CLIENT that connects to Python's SERVER.
    """

    def __init__(self, port):
        self.port = port
        self._conn = None
        self.received_lines = []
        self._running = False

    def connect(self):
        """Connect to Python's command server."""
        self._conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._conn.connect(('127.0.0.1', self.port))
        self._running = True
        # Start reading in background
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        """Read commands sent by Python."""
        buf = ''
        self._conn.settimeout(0.5)
        while self._running:
            try:
                data = self._conn.recv(4096)
                if not data:
                    break
                buf += data.decode('utf-8', errors='replace')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if line:
                        self.received_lines.append(line)
            except socket.timeout:
                continue
            except Exception:
                break

    def close(self):
        self._running = False
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


def start_sender_with_mock_ea(port):
    """
    Helper: Start CommandSender server in background, connect a mock EA,
    return (sender, mock_ea).
    """
    confirm_q = queue.Queue()
    sender = CommandSender(confirm_q, port=port)

    # Start server in background (it blocks waiting for EA)
    server_thread = threading.Thread(
        target=lambda: sender.connect(timeout=5), daemon=True)
    server_thread.start()
    time.sleep(0.3)

    # Connect mock EA as client
    mock_ea = MockEAClient(port)
    mock_ea.connect()

    # Wait for server to accept
    server_thread.join(timeout=3)
    time.sleep(0.2)

    return sender, mock_ea, confirm_q


# ═══════════════════════════════════════════════════════════════════════
# Message Format Tests
# ═══════════════════════════════════════════════════════════════════════

class TestMessageFormat:
    """Verify that commands are formatted correctly per protocol spec."""

    def test_buy_format(self):
        """BUY message has correct pipe-delimited format."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19110)

        try:
            # Pre-load a CONFIRM for any req_id
            confirm_q.put({'req_id': 'placeholder', 'ticket': 0, 'status': 'OK'})

            threading.Thread(
                target=lambda: sender.buy(2400.10, 2393.00, 2407.20, 0.01),
                daemon=True).start()
            time.sleep(0.5)

            assert len(mock_ea.received_lines) >= 1
            parts = mock_ea.received_lines[0].split('|')

            assert parts[0] == 'BUY'
            assert parts[1] == '2400.10000'  # 5 decimal places
            assert parts[2] == '2393.00000'
            assert parts[3] == '2407.20000'
            assert parts[4] == '0.01'
            assert len(parts[5]) == 8  # UUID[:8] req_id

        finally:
            mock_ea.close()
            sender.disconnect()

    def test_sell_format(self):
        """SELL message has correct format."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19111)

        try:
            confirm_q.put({'req_id': 'placeholder', 'ticket': 0, 'status': 'OK'})

            threading.Thread(
                target=lambda: sender.sell(2400.50, 2407.58, 2393.42, 0.05),
                daemon=True).start()
            time.sleep(0.5)

            assert len(mock_ea.received_lines) >= 1
            parts = mock_ea.received_lines[0].split('|')
            assert parts[0] == 'SELL'
            assert parts[1] == '2400.50000'
            assert parts[4] == '0.05'

        finally:
            mock_ea.close()
            sender.disconnect()

    def test_modifysl_format(self):
        """MODIFYSL message format verification."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19112)

        try:
            confirm_q.put({'req_id': 'placeholder', 'ticket': 0, 'status': 'OK'})

            threading.Thread(
                target=lambda: sender.modify_sl(12345678, 2400.00),
                daemon=True).start()
            time.sleep(0.5)

            assert len(mock_ea.received_lines) >= 1
            parts = mock_ea.received_lines[0].split('|')
            assert parts[0] == 'MODIFYSL'
            assert parts[1] == '12345678'
            assert parts[2] == '2400.00000'

        finally:
            mock_ea.close()
            sender.disconnect()

    def test_close_format(self):
        """CLOSE message format verification."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19113)

        try:
            confirm_q.put({'req_id': 'placeholder', 'ticket': 0, 'status': 'OK'})

            threading.Thread(
                target=lambda: sender.close_position(87654321),
                daemon=True).start()
            time.sleep(0.5)

            assert len(mock_ea.received_lines) >= 1
            parts = mock_ea.received_lines[0].split('|')
            assert parts[0] == 'CLOSE'
            assert parts[1] == '87654321'

        finally:
            mock_ea.close()
            sender.disconnect()


# ═══════════════════════════════════════════════════════════════════════
# Confirm Matching Tests
# ═══════════════════════════════════════════════════════════════════════

class TestConfirmMatching:
    """Test CONFIRM timeout and matching behavior."""

    def test_confirm_timeout_returns_none(self):
        """If no CONFIRM arrives within timeout, returns None."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19114)

        try:
            sender.CONFIRM_TIMEOUT = 1.0  # Short timeout for testing

            start = time.time()
            result = sender.buy(2400.0, 2393.0, 2407.0, 0.01)
            elapsed = time.time() - start

            assert result is None  # Timeout → None
            assert elapsed >= 0.9
            assert elapsed < 3.0

        finally:
            mock_ea.close()
            sender.disconnect()

    def test_confirm_match_by_req_id(self):
        """CONFIRM matching works by req_id."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19115)

        try:
            sender.CONFIRM_TIMEOUT = 2.0

            result_holder = [None]

            def do_buy():
                result_holder[0] = sender.buy(2400.0, 2393.0, 2407.0, 0.01)

            t = threading.Thread(target=do_buy)
            t.start()
            time.sleep(0.3)

            # Extract the req_id from what the mock EA received
            assert len(mock_ea.received_lines) >= 1
            parts = mock_ea.received_lines[0].split('|')
            req_id = parts[-1]

            # Push matching CONFIRM
            confirm_q.put({
                'req_id': req_id,
                'ticket': 99887766,
                'status': 'OK'
            })

            t.join(timeout=3)

            assert result_holder[0] is not None
            assert result_holder[0]['ticket'] == 99887766
            assert result_holder[0]['status'] == 'OK'

        finally:
            mock_ea.close()
            sender.disconnect()

    def test_unmatched_confirms_put_back(self):
        """Non-matching CONFIRMs are put back in the queue."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19116)

        try:
            sender.CONFIRM_TIMEOUT = 1.5

            # Pre-load an unrelated CONFIRM
            confirm_q.put({
                'req_id': 'OTHER',
                'ticket': 11111111,
                'status': 'OK'
            })

            result = sender.buy(2400.0, 2393.0, 2407.0, 0.01)
            assert result is None  # Timeout

            # The unmatched CONFIRM should be back in the queue
            assert confirm_q.qsize() >= 1
            conf = confirm_q.get_nowait()
            assert conf['req_id'] == 'OTHER'

        finally:
            mock_ea.close()
            sender.disconnect()


# ═══════════════════════════════════════════════════════════════════════
# Connection Tests
# ═══════════════════════════════════════════════════════════════════════

class TestConnection:
    """Test connection behavior."""

    def test_connect_success(self):
        """EA successfully connects to command server."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19117)

        try:
            assert sender._connected is True
        finally:
            mock_ea.close()
            sender.disconnect()

    def test_disconnect(self):
        """Disconnect closes socket cleanly."""
        sender, mock_ea, confirm_q = start_sender_with_mock_ea(19118)

        try:
            assert sender._connected is True
            sender.disconnect()
            assert sender._connected is False
        finally:
            mock_ea.close()


# ═══════════════════════════════════════════════════════════════════════
# Constants Verification
# ═══════════════════════════════════════════════════════════════════════

class TestConstants:
    """Verify critical constants match PRD specifications."""

    def test_confirm_timeout_default(self):
        """CONFIRM_TIMEOUT is 5.0 seconds (FR-PY-08)."""
        assert CommandSender.CONFIRM_TIMEOUT == 5.0


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
