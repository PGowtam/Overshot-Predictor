"""
BrickOfTicks Socket Bridge — Command Sender
=============================================
TCP SERVER on port 9001 that accepts the EA's command connection,
then sends trade commands to the MT5 TickSender EA.

Architecture:
  The EA connects to Python as a CLIENT on port 9001 (same as port 9000).
  Python acts as SERVER on both ports. This is the "dual-client EA" design
  that avoids DLL/SocketBind issues on the MQL5 side under Wine/Mac.

Protocol Messages (newline-terminated, pipe-delimited):
  BUY|<price>|<sl>|<tp>|<volume>|<req_id>
  SELL|<price>|<sl>|<tp>|<volume>|<req_id>
  CLOSE|<ticket>|<req_id>
  MODIFYSL|<ticket>|<new_sl>|<req_id>

CONFIRM timeout: 5 seconds hard limit. NO auto-retry on timeout.
Security: Binds ONLY to 127.0.0.1 (localhost). Never 0.0.0.0.
"""

import socket
import threading
import uuid
import logging
import queue
import time

logger = logging.getLogger(__name__)


class CommandSender:
    """
    TCP server on port 9001 that accepts the EA's command connection,
    then sends trade commands and waits for CONFIRM via the tick channel.

    Architecture:
      Python LISTENS on port 9001 → EA CONNECTS as client → Python sends
      commands over the accepted connection.

    Attributes:
        CONFIRM_TIMEOUT: 5-second hard timeout for trade confirmations.
    """

    CONFIRM_TIMEOUT = 5.0  # seconds — HARD REQUIREMENT per FR-PY-08

    def __init__(self, confirm_queue, port=9001):
        """
        Args:
            confirm_queue: queue.Queue shared with TickReceiver for CONFIRM messages.
            port: Port to listen on (9001 by default).
        """
        self._port = port
        self._server = None
        self._conn = None
        self._confirms = confirm_queue
        self._connected = False

    def connect(self, timeout=30):
        """
        Start TCP server on port 9001 and wait for EA to connect.

        The EA connects as a client to this port during its OnInit().
        This method blocks until the EA connects or timeout expires.
        """
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # CRITICAL: bind to 127.0.0.1 ONLY — never 0.0.0.0
        self._server.bind(('127.0.0.1', self._port))
        self._server.listen(1)
        self._server.settimeout(timeout)
        logger.info(f"CommandSender listening on 127.0.0.1:{self._port} — waiting for EA...")

        try:
            self._conn, addr = self._server.accept()
            self._connected = True
            logger.info(f"CommandSender: EA connected from {addr}")
        except socket.timeout:
            self._connected = False
            logger.error(f"CommandSender: No EA connection within {timeout}s")
            raise ConnectionRefusedError(
                f"No EA connection on port {self._port} within {timeout}s")

    def disconnect(self):
        """Gracefully close the command connection and server."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        self._connected = False

    def _send(self, msg_type, fields):
        """
        Format and send a pipe-delimited command with a unique request ID.

        Returns the req_id string, or None if send failed.
        """
        req_id = str(uuid.uuid4())[:8]
        line = '|'.join([msg_type] + [str(f) for f in fields] + [req_id]) + '\n'

        try:
            self._conn.sendall(line.encode('utf-8'))
            logger.info(f"SENT: {line.strip()}")
            return req_id
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.error(f"CommandSender: send failed ({e}) — attempting to accept new connection")
            try:
                # EA may have reconnected — try accepting a new connection
                self._server.settimeout(5)
                self._conn, addr = self._server.accept()
                self._connected = True
                logger.info(f"CommandSender: EA reconnected from {addr}")
                self._conn.sendall(line.encode('utf-8'))
                logger.info(f"SENT (after reconnect): {line.strip()}")
                return req_id
            except Exception as e2:
                logger.critical(f"CommandSender: reconnect failed ({e2}) — entering degraded mode")
                self._connected = False
                return None

    def _await_confirm(self, req_id):
        """
        Wait up to CONFIRM_TIMEOUT (5s) for a matching CONFIRM message.

        CONFIRMs arrive on the TICK channel (port 9000) via the shared
        confirm_queue from TickReceiver.

        Returns:
            dict with keys {req_id, ticket, status} on success.
            None on timeout (operator must check MT5 terminal manually).
        """
        if req_id is None:
            return None

        deadline = time.time() + self.CONFIRM_TIMEOUT
        unmatched = []

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                conf = self._confirms.get(timeout=min(0.1, remaining))
                if conf['req_id'] == req_id:
                    # Put back any unmatched confirms we collected
                    for u in unmatched:
                        self._confirms.put(u)
                    return conf
                else:
                    unmatched.append(conf)
            except queue.Empty:
                continue

        # Put back unmatched confirms
        for u in unmatched:
            self._confirms.put(u)

        logger.error(f"COMMAND TIMEOUT — req_id={req_id}. "
                     f"ACTION REQUIRED: Check MT5 terminal manually for pending orders.")
        return None

    def buy(self, price, sl, tp, volume=0.01):
        """
        Send BUY command and wait for CONFIRM.

        Args:
            price: Entry price (5 decimal places)
            sl: Stop loss price
            tp: Take profit price
            volume: Lot size (default 0.01)

        Returns:
            CONFIRM dict or None on timeout.
        """
        req_id = self._send('BUY', [
            f'{price:.5f}', f'{sl:.5f}', f'{tp:.5f}', f'{volume:.2f}'
        ])
        return self._await_confirm(req_id)

    def sell(self, price, sl, tp, volume=0.01):
        """
        Send SELL command and wait for CONFIRM.

        Returns:
            CONFIRM dict or None on timeout.
        """
        req_id = self._send('SELL', [
            f'{price:.5f}', f'{sl:.5f}', f'{tp:.5f}', f'{volume:.2f}'
        ])
        return self._await_confirm(req_id)

    def modify_sl(self, ticket, new_sl):
        """
        Send MODIFYSL command and wait for CONFIRM.

        Returns:
            CONFIRM dict or None on timeout.
        """
        req_id = self._send('MODIFYSL', [ticket, f'{new_sl:.5f}'])
        return self._await_confirm(req_id)

    def close_position(self, ticket):
        """
        Send CLOSE command and wait for CONFIRM.

        Returns:
            CONFIRM dict or None on timeout.
        """
        req_id = self._send('CLOSE', [ticket])
        return self._await_confirm(req_id)
