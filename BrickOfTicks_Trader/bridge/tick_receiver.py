"""
BrickOfTicks Socket Bridge — Tick Receiver
==========================================
TCP server on port 9000 that receives tick data from MT5 TickSender EA.

Protocol Messages (newline-terminated, pipe-delimited):
  DAYOPEN|<time_msc>|<price>
  HTICK|<time_msc>|<bid>|<ask>|<bid_vol>|<ask_vol>
  HDONE|<count>
  TICK|<time_msc>|<bid>|<ask>|<bid_vol>|<ask_vol>
  HEARTBEAT|<time_msc>
  CONFIRM|<req_id>|<ticket>|OK   or  CONFIRM|<req_id>|0|ERROR|<code>

Security: Binds ONLY to 127.0.0.1 (localhost). Never 0.0.0.0.
"""

import socket
import threading
import queue
import logging
import time

logger = logging.getLogger(__name__)


class TickReceiver:
    """
    TCP server that receives tick data and trade confirmations from MT5 EA.

    Attributes:
        tick_queue:      Thread-safe queue of live tick dicts (maxsize=10000)
        history_ticks:   List of history tick dicts replayed for warmup
        history_done:    Event set when HDONE is received
        confirm_queue:   Queue of CONFIRM messages for CommandSender
        day_open_price:  Latest daily open price from DAYOPEN message
        tick_count:      Total live ticks received (for audit comparison)
    """

    def __init__(self, port=9000):
        self.tick_queue = queue.Queue(maxsize=10000)
        self.history_ticks = []
        self.history_done = threading.Event()
        self.confirm_queue = queue.Queue()
        self.day_open_price = None
        self.day_open_time = None
        self._port = port
        self._server = None
        self._conn = None
        self._running = False
        self.tick_count = 0
        self.htick_count = 0
        self._lock = threading.Lock()

    def start(self):
        """Start TCP server on localhost:9000 and spawn accept thread."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # CRITICAL: bind to 127.0.0.1 ONLY — never 0.0.0.0
        self._server.bind(('127.0.0.1', self._port))
        self._server.listen(1)
        self._running = True
        logger.info(f"TickReceiver listening on 127.0.0.1:{self._port}")
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()

    def stop(self):
        """Gracefully stop the receiver."""
        self._running = False
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass

    def _accept_loop(self, srv=None):
        """Accept one persistent connection from EA and read messages."""
        server = srv or self._server
        try:
            server.settimeout(1.0)
            while self._running:
                try:
                    conn, addr = server.accept()
                    self._conn = conn
                    logger.info(f"EA connected from {addr}")
                    self._read_loop(conn)
                except socket.timeout:
                    continue
                except OSError:
                    break
        except Exception as e:
            logger.error(f"Accept loop error: {e}")

    def _read_loop(self, conn):
        """Read from connection, split on newlines, dispatch each line."""
        buf = ''
        conn.settimeout(1.0)
        while self._running:
            try:
                data = conn.recv(4096)
                if not data:
                    logger.warning("EA connection closed (recv returned empty)")
                    break
                buf += data.decode('utf-8', errors='replace')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if line:
                        self._dispatch(line)
            except socket.timeout:
                continue
            except ConnectionResetError:
                logger.warning("EA connection reset")
                break
            except Exception as e:
                logger.error(f"Read loop error: {e}")
                break

    def _dispatch(self, line):
        """Parse and route a single protocol message."""
        try:
            parts = line.split('|')
            msg_type = parts[0]

            if msg_type == 'TICK':
                tick = {
                    'time_msc': int(parts[1]),
                    'bid': float(parts[2]),
                    'ask': float(parts[3]),
                    'bid_vol': float(parts[4]),
                    'ask_vol': float(parts[5])
                }
                try:
                    self.tick_queue.put_nowait(tick)
                except queue.Full:
                    # Drop oldest tick to make room
                    try:
                        self.tick_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.tick_queue.put_nowait(tick)
                    logger.warning("TICK BUFFER OVERFLOW — dropping oldest tick")
                with self._lock:
                    self.tick_count += 1

            elif msg_type == 'HTICK':
                htick = {
                    'time_msc': int(parts[1]),
                    'bid': float(parts[2]),
                    'ask': float(parts[3]),
                    'bid_vol': float(parts[4]),
                    'ask_vol': float(parts[5])
                }
                with self._lock:
                    self.history_ticks.append(htick)
                    self.htick_count += 1

            elif msg_type == 'HDONE':
                count = int(parts[1])
                logger.info(f"History batch complete: {count} ticks (received {self.htick_count})")
                self.history_done.set()

            elif msg_type == 'DAYOPEN':
                self.day_open_time = int(parts[1])
                self.day_open_price = float(parts[2])
                logger.info(f"Day open price: {self.day_open_price}")

            elif msg_type == 'CONFIRM':
                confirm = {
                    'req_id': parts[1],
                    'ticket': int(parts[2]),
                    'status': parts[3]
                }
                # If ERROR, include error code if present
                if len(parts) > 4:
                    confirm['error_code'] = parts[4]
                self.confirm_queue.put(confirm)
                logger.info(f"CONFIRM received: req_id={confirm['req_id']} "
                            f"ticket={confirm['ticket']} status={confirm['status']}")

            elif msg_type == 'HEARTBEAT':
                # No-op — just prevents stall detection
                pass

            else:
                logger.warning(f"Unknown message type: {msg_type} — full line: {line[:100]}")

        except (IndexError, ValueError) as e:
            logger.error(f"Failed to parse message: {line[:100]} — error: {e}")

    def feed_line(self, line):
        """
        Feed a raw protocol line for testing/simulation.
        Useful for unit tests without needing a real socket connection.
        """
        self._dispatch(line.strip())

    def get_history_ticks(self):
        """Thread-safe snapshot copy of history ticks."""
        with self._lock:
            return list(self.history_ticks)

    def set_history_ticks(self, ticks):
        """Thread-safe setter for history ticks."""
        with self._lock:
            self.history_ticks = list(ticks)
            self.htick_count = len(ticks)
