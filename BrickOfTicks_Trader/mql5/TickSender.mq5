//+------------------------------------------------------------------+
//|                                                   TickSender.mq5 |
//|                                  Copyright 2026, BrickOfTicks    |
//|                                             https://localhost    |
//+------------------------------------------------------------------+
//| Production TCP Socket Bridge EA v3.0                             |
//|                                                                  |
//| Architecture:                                                    |
//|   Port 9000: EA → Python  (tick data + confirmations)            |
//|   Port 9001: EA ← Python  (trade commands)                       |
//|                                                                  |
//| Both sockets operate as CLIENTS connecting to Python servers.     |
//| This avoids DLL requirements and works cleanly under Wine/Mac.   |
//|                                                                  |
//| Protocol: UTF-8, newline-terminated, pipe-delimited              |
//| K = 0.00295 | Pred_OS >= 1.4 | Baiting: DISABLED                |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, BrickOfTicks"
#property link      "https://localhost"
#property version   "3.00"
#property strict

//--- Input parameters
input int    InpTickPort       = 9000;     // Tick channel port (EA → Python)
input int    InpCmdPort        = 9001;     // Command channel port (Python → EA)
input int    InpHistoryTicks   = 5000;     // Number of history ticks to send on startup
input int    InpConnectRetries = 10;       // Max connection retry attempts
input int    InpConnectWaitMs  = 2000;     // Wait between retries (ms)
input int    InpTimerMs        = 100;      // Command poll interval (ms)
input int    InpMagicNumber    = 314159;   // Magic number for orders
input int    InpSlippage       = 20;       // Max slippage in points
input double InpDefaultVolume  = 0.01;     // Default lot size

//--- Socket handles
int tick_socket = INVALID_HANDLE;
int cmd_socket  = INVALID_HANDLE;

//--- State tracking
string last_day_open_date = "";
string cmd_buffer         = "";
bool   tick_connected     = false;
bool   cmd_connected      = false;
bool   reconnecting       = false;  // Guard: prevents SendHistory cascade
int    tick_send_count     = 0;
int    cmd_recv_count      = 0;
datetime last_heartbeat    = 0;
datetime last_tick_time    = 0;
datetime last_cmd_retry    = 0;     // Last cmd socket reconnect attempt

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("═══════════════════════════════════════════════════");
   Print("  TickSender EA v3.00 — Production Socket Bridge");
   Print("  Tick Port: ", InpTickPort, " | Cmd Port: ", InpCmdPort);
   Print("  History:   ", InpHistoryTicks, " ticks");
   Print("═══════════════════════════════════════════════════");

   // ─── Step 1: Connect to Python tick receiver (port 9000) ─────
   tick_socket = SocketCreate();
   if(tick_socket == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create tick socket. Error: ", GetLastError());
      return INIT_FAILED;
   }

   tick_connected = false;
   for(int i = 0; i < InpConnectRetries; i++)
   {
      if(SocketConnect(tick_socket, "127.0.0.1", InpTickPort, InpConnectWaitMs))
      {
         tick_connected = true;
         Print("✓ Tick socket connected to 127.0.0.1:", InpTickPort,
               " (attempt ", i+1, "/", InpConnectRetries, ")");
         break;
      }
      Print("  Tick socket retry ", i+1, "/", InpConnectRetries,
            " — waiting ", InpConnectWaitMs, "ms...");
      Sleep(InpConnectWaitMs);
   }

   if(!tick_connected)
   {
      Print("CRITICAL: Failed to connect tick socket after ",
            InpConnectRetries, " attempts.");
      Print("  → Is Python bridge running? (python bridge/main.py)");
      return INIT_FAILED;
   }

   // ─── Step 2: Send DAYOPEN ────────────────────────────────────
   double day_open = iOpen(_Symbol, PERIOD_D1, 0);
   long   time_msc = (long)TimeCurrent() * 1000;  // Approximate ms
   
   // Try to get more precise time from last tick
   MqlTick last_tick;
   if(SymbolInfoTick(_Symbol, last_tick))
      time_msc = last_tick.time_msc;

   string dayopen_msg = StringFormat("DAYOPEN|%lld|%.5f", time_msc, day_open);
   if(!SendTickMsg(dayopen_msg))
   {
      Print("ERROR: Failed to send DAYOPEN message");
      return INIT_FAILED;
   }
   Print("✓ Sent DAYOPEN: price=", DoubleToString(day_open, 5));
   last_day_open_date = TimeToString(TimeCurrent(), TIME_DATE);

   // ─── Step 3: Send history batch ─────────────────────────────
   int history_sent = SendHistory(InpHistoryTicks);
   Print("✓ Sent history: ", history_sent, " ticks + HDONE");

   // ─── Step 4: Connect to Python command receiver (port 9001) ──
   cmd_socket = SocketCreate();
   if(cmd_socket == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create command socket. Error: ", GetLastError());
      return INIT_FAILED;
   }

   cmd_connected = false;
   for(int i = 0; i < InpConnectRetries; i++)
   {
      if(SocketConnect(cmd_socket, "127.0.0.1", InpCmdPort, InpConnectWaitMs))
      {
         cmd_connected = true;
         Print("✓ Command socket connected to 127.0.0.1:", InpCmdPort,
               " (attempt ", i+1, "/", InpConnectRetries, ")");
         break;
      }
      Print("  Cmd socket retry ", i+1, "/", InpConnectRetries,
            " — waiting ", InpConnectWaitMs, "ms...");
      Sleep(InpConnectWaitMs);
   }

   if(!cmd_connected)
   {
      Print("WARNING: Command socket not connected. Orders will fail.");
      Print("  → Python bridge command server may not be ready yet.");
      // Don't fail init — tick streaming can still work for Phase -1 audit
   }

   // ─── Step 5: Start timer for command polling ────────────────
   EventSetMillisecondTimer(InpTimerMs);
   last_heartbeat = TimeCurrent();
   last_tick_time = TimeCurrent();

   Print("═══════════════════════════════════════════════════");
   Print("  TickSender EA INITIALIZED SUCCESSFULLY");
   Print("  Streaming live ticks on port ", InpTickPort);
   if(cmd_connected)
      Print("  Listening for commands on port ", InpCmdPort);
   Print("═══════════════════════════════════════════════════");

   return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("TickSender shutting down. Reason: ", reason);
   Print("  Total ticks sent: ", tick_send_count);
   Print("  Total commands received: ", cmd_recv_count);

   // Close all socket handles gracefully
   if(tick_socket != INVALID_HANDLE)
   {
      SocketClose(tick_socket);
      tick_socket = INVALID_HANDLE;
   }
   if(cmd_socket != INVALID_HANDLE)
   {
      SocketClose(cmd_socket);
      cmd_socket = INVALID_HANDLE;
   }

   EventKillTimer();
   Print("TickSender shutdown complete.");
}


//+------------------------------------------------------------------+
//| Expert tick function — fires on EVERY price change               |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!tick_connected) return;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   // ─── Send TICK message ───────────────────────────────────────
   // Format: TICK|<time_msc>|<bid>|<ask>|<bid_vol>|<ask_vol>
   // Note: volume_real is used for both bid_vol and ask_vol
   //       (MT5 does not distinguish bid/ask volume at tick level)
   string msg = StringFormat("TICK|%lld|%.5f|%.5f|%.2f|%.2f",
      tick.time_msc, tick.bid, tick.ask,
      tick.volume_real, tick.volume_real);

   if(SendTickMsg(msg))
   {
      tick_send_count++;
      last_tick_time = TimeCurrent();
   }

   // ─── Daily rollover detection ────────────────────────────────
   string today = TimeToString(TimeCurrent(), TIME_DATE);
   if(today != last_day_open_date)
   {
      double new_open = iOpen(_Symbol, PERIOD_D1, 0);
      string dayopen_msg = StringFormat("DAYOPEN|%lld|%.5f",
         tick.time_msc, new_open);

      if(SendTickMsg(dayopen_msg))
      {
         Print("ROLLOVER detected: new date=", today,
               " open=", DoubleToString(new_open, 5));
         last_day_open_date = today;
      }
   }
}


//+------------------------------------------------------------------+
//| Timer function — polls for commands every 100ms                  |
//+------------------------------------------------------------------+
void OnTimer()
{
   // ─── Heartbeat: send if no tick for 500ms ────────────────────
   if(tick_connected && (TimeCurrent() - last_tick_time >= 1))
   {
      // Only send heartbeat at most every 1 second
      if(TimeCurrent() - last_heartbeat >= 1)
      {
         long now_msc = (long)TimeCurrent() * 1000;
         SendTickMsg(StringFormat("HEARTBEAT|%lld", now_msc));
         last_heartbeat = TimeCurrent();
      }
   }

   // ─── Command socket reconnect ─────────────────────────────────
   // If cmd_socket disconnected (e.g. Python restarted), try to reconnect
   // every 5 seconds. This handles the case where Python was restarted
   // after the EA's OnInit() already ran.
   if(!cmd_connected && (TimeCurrent() - last_cmd_retry >= 5))
   {
      last_cmd_retry = TimeCurrent();
      Print("Attempting cmd socket reconnect to 127.0.0.1:", InpCmdPort, "...");

      if(cmd_socket != INVALID_HANDLE)
         SocketClose(cmd_socket);

      cmd_socket = SocketCreate();
      if(cmd_socket != INVALID_HANDLE &&
         SocketConnect(cmd_socket, "127.0.0.1", InpCmdPort, 1000))
      {
         cmd_connected = true;
         cmd_buffer = "";
         Print("✓ Command socket reconnected to 127.0.0.1:", InpCmdPort);
      }
   }

   // ─── Read commands from Python ───────────────────────────────
   if(!cmd_connected) return;

   // WINE/macOS FIX: SocketRead with timeout fails under Wine when
   // no data is available (Error 5273: IO Error). Use SocketIsReadable()
   // to check if data exists BEFORE attempting to read. This completely
   // avoids the broken timeout behavior under Wine.
   uint readable = SocketIsReadable(cmd_socket);

   if(readable == 0)
      return;  // No data available — just return, don't try to read

   uchar recv_buf[];
   int bytes_read = SocketRead(cmd_socket, recv_buf, (int)MathMin(readable, 1024), 100);

   if(bytes_read <= 0)
   {
      int err = GetLastError();
      if(err != 0 && err != 5273)  // Ignore transient IO errors
      {
         Print("Command socket read error: ", err, " — marking disconnected");
         cmd_connected = false;
      }
      return;
   }

   string received = CharArrayToString(recv_buf, 0, bytes_read, CP_UTF8);
   cmd_buffer += received;

   // Process complete lines
   while(StringFind(cmd_buffer, "\n") >= 0)
   {
      int nl_pos = StringFind(cmd_buffer, "\n");
      string line = StringSubstr(cmd_buffer, 0, nl_pos);
      cmd_buffer = StringSubstr(cmd_buffer, nl_pos + 1);

      // Trim whitespace
      StringTrimLeft(line);
      StringTrimRight(line);

      if(StringLen(line) > 0)
      {
         ProcessCommand(line);
         cmd_recv_count++;
      }
   }
}


//+------------------------------------------------------------------+
//| Process a single command from Python                             |
//+------------------------------------------------------------------+
void ProcessCommand(string line)
{
   Print("CMD received: ", line);

   string parts[];
   int n = StringSplit(line, '|', parts);
   if(n < 2)
   {
      Print("ERROR: Invalid command format — too few fields: ", line);
      return;
   }

   string cmd_type = parts[0];
   string req_id = parts[n - 1];  // Last field is always req_id

   // ─── BUY / SELL ──────────────────────────────────────────────
   if(cmd_type == "BUY" || cmd_type == "SELL")
   {
      if(n < 6)
      {
         Print("ERROR: BUY/SELL requires 6 fields, got ", n);
         SendConfirm(req_id, 0, "ERROR|INVALID_FORMAT");
         return;
      }

      double price  = StringToDouble(parts[1]);
      double sl     = StringToDouble(parts[2]);
      double tp     = StringToDouble(parts[3]);
      double volume = StringToDouble(parts[4]);

      MqlTradeRequest request = {};
      MqlTradeResult  result  = {};

      request.action    = TRADE_ACTION_DEAL;
      request.symbol    = _Symbol;
      request.volume    = volume;
      request.price     = (cmd_type == "BUY") ?
                           SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                           SymbolInfoDouble(_Symbol, SYMBOL_BID);
      request.sl        = sl;
      request.tp        = tp;
      request.type      = (cmd_type == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      request.magic     = InpMagicNumber;
      request.deviation = InpSlippage;
      request.comment   = "BrickOfTicks|" + req_id;

      // Try IOC first, then FOK if IOC not supported
      request.type_filling = ORDER_FILLING_IOC;

      bool sent = OrderSend(request, result);

      if(sent && result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✓ Order executed: ", cmd_type,
               " ticket=", result.order,
               " price=", DoubleToString(result.price, 5),
               " vol=", DoubleToString(volume, 2));
         SendConfirm(req_id, (int)result.order, "OK");
      }
      else
      {
         // Try FOK filling if IOC failed
         if(result.retcode == TRADE_RETCODE_INVALID_FILL)
         {
            request.type_filling = ORDER_FILLING_FOK;
            sent = OrderSend(request, result);
            if(sent && result.retcode == TRADE_RETCODE_DONE)
            {
               Print("✓ Order executed (FOK): ", cmd_type,
                     " ticket=", result.order);
               SendConfirm(req_id, (int)result.order, "OK");
               return;
            }
         }

         Print("✗ Order FAILED: ", cmd_type,
               " retcode=", result.retcode,
               " comment=", result.comment);
         SendConfirm(req_id, 0,
            StringFormat("ERROR|%d", result.retcode));
      }
   }
   // ─── CLOSE ───────────────────────────────────────────────────
   else if(cmd_type == "CLOSE")
   {
      if(n < 3)
      {
         Print("ERROR: CLOSE requires 3 fields, got ", n);
         SendConfirm(req_id, 0, "ERROR|INVALID_FORMAT");
         return;
      }

      ulong ticket = (ulong)StringToInteger(parts[1]);

      // Select the position
      if(!PositionSelectByTicket(ticket))
      {
         Print("ERROR: Position not found for ticket ", ticket);
         SendConfirm(req_id, (int)ticket, "ERROR|POSITION_NOT_FOUND");
         return;
      }

      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      MqlTradeRequest request = {};
      MqlTradeResult  result  = {};

      request.action   = TRADE_ACTION_DEAL;
      request.symbol   = _Symbol;
      request.volume   = volume;
      request.position = ticket;
      request.price    = (pos_type == POSITION_TYPE_BUY) ?
                          SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                          SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      request.type     = (pos_type == POSITION_TYPE_BUY) ?
                          ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      request.magic    = InpMagicNumber;
      request.deviation = InpSlippage;
      request.type_filling = ORDER_FILLING_IOC;

      bool sent = OrderSend(request, result);
      if(sent && result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✓ Position closed: ticket=", ticket);
         SendConfirm(req_id, (int)ticket, "OK");
      }
      else
      {
         Print("✗ Close FAILED: ticket=", ticket,
               " retcode=", result.retcode);
         SendConfirm(req_id, (int)ticket,
            StringFormat("ERROR|%d", result.retcode));
      }
   }
   // ─── MODIFYSL ────────────────────────────────────────────────
   else if(cmd_type == "MODIFYSL")
   {
      if(n < 4)
      {
         Print("ERROR: MODIFYSL requires 4 fields, got ", n);
         SendConfirm(req_id, 0, "ERROR|INVALID_FORMAT");
         return;
      }

      ulong  ticket = (ulong)StringToInteger(parts[1]);
      double new_sl = StringToDouble(parts[2]);

      // Select the position to get current TP
      if(!PositionSelectByTicket(ticket))
      {
         Print("ERROR: Position not found for ticket ", ticket);
         SendConfirm(req_id, (int)ticket, "ERROR|POSITION_NOT_FOUND");
         return;
      }

      double current_tp = PositionGetDouble(POSITION_TP);

      MqlTradeRequest request = {};
      MqlTradeResult  result  = {};

      request.action   = TRADE_ACTION_SLTP;
      request.symbol   = _Symbol;
      request.position = ticket;
      request.sl       = new_sl;
      request.tp       = current_tp;  // Preserve existing TP

      bool sent = OrderSend(request, result);
      if(sent && result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✓ SL modified: ticket=", ticket,
               " new_sl=", DoubleToString(new_sl, 5));
         SendConfirm(req_id, (int)ticket, "OK");
      }
      else
      {
         Print("✗ MODIFYSL FAILED: ticket=", ticket,
               " retcode=", result.retcode);
         SendConfirm(req_id, (int)ticket,
            StringFormat("ERROR|%d", result.retcode));
      }
   }
   else
   {
      Print("WARNING: Unknown command type: ", cmd_type);
      SendConfirm(req_id, 0, "ERROR|UNKNOWN_COMMAND");
   }
}


//+------------------------------------------------------------------+
//| Send history ticks for warmup                                    |
//+------------------------------------------------------------------+
int SendHistory(int count)
{
   MqlTick ticks[];
   int copied = CopyTicks(_Symbol, ticks, COPY_TICKS_ALL, 0, count);

   if(copied <= 0)
   {
      Print("WARNING: CopyTicks returned ", copied, " ticks");
      // Send HDONE with 0 count
      SendTickMsg(StringFormat("HDONE|%d", 0));
      return 0;
   }

   Print("Sending ", copied, " history ticks...");

   int sent = 0;
   for(int i = 0; i < copied; i++)
   {
      string msg = StringFormat("HTICK|%lld|%.5f|%.5f|%.2f|%.2f",
         ticks[i].time_msc, ticks[i].bid, ticks[i].ask,
         ticks[i].volume_real, ticks[i].volume_real);

      if(SendTickMsg(msg))
         sent++;

      // Yield every 500 ticks to prevent socket buffer overflow
      if(i > 0 && i % 500 == 0)
         Sleep(10);
   }

   // Send HDONE marker
   SendTickMsg(StringFormat("HDONE|%d", sent));
   return sent;
}


//+------------------------------------------------------------------+
//| Send a message on the tick socket (port 9000)                    |
//+------------------------------------------------------------------+
bool SendTickMsg(string msg)
{
   if(tick_socket == INVALID_HANDLE || !tick_connected)
      return false;

   // Ensure newline termination
   if(StringFind(msg, "\n") < 0)
      msg += "\n";

   uchar data[];
   int len = StringToCharArray(msg, data, 0, -1, CP_UTF8);
   // StringToCharArray adds null terminator; don't send it
   if(len > 0) len--;

   int bytes_sent = SocketSend(tick_socket, data, len);

   if(bytes_sent <= 0)
   {
      int err = GetLastError();
      Print("ERROR: SocketSend failed. Error: ", err);

      // Connection lost — attempt reconnect
      tick_connected = false;
      Print("Tick socket disconnected. Attempting reconnect...");

      SocketClose(tick_socket);
      tick_socket = SocketCreate();

      if(tick_socket != INVALID_HANDLE &&
         SocketConnect(tick_socket, "127.0.0.1", InpTickPort, InpConnectWaitMs))
      {
         tick_connected = true;
         Print("✓ Tick socket reconnected");

         // Re-send DAYOPEN
         double day_open = iOpen(_Symbol, PERIOD_D1, 0);
         long time_msc = (long)TimeCurrent() * 1000;
         string dayopen = StringFormat("DAYOPEN|%lld|%.5f\n", time_msc, day_open);
         uchar d2[];
         int l2 = StringToCharArray(dayopen, d2, 0, -1, CP_UTF8);
         if(l2 > 0) l2--;
         SocketSend(tick_socket, d2, l2);

         // Re-send history — but ONLY if not already inside a reconnect
         // (prevents SendHistory → SendTickMsg → reconnect → SendHistory cascade)
         if(!reconnecting)
         {
            reconnecting = true;
            int hist_sent = SendHistory(InpHistoryTicks);
            reconnecting = false;
            Print("✓ Reconnect: re-sent DAYOPEN + ", hist_sent, " history ticks + HDONE");
         }
         else
         {
            Print("  (skipping history replay — already inside reconnect)");
         }

         // Also mark cmd as disconnected so it reconnects via OnTimer
         cmd_connected = false;

         return true;
      }
      else
      {
         Print("CRITICAL: Tick socket reconnect failed");
         return false;
      }
   }

   return true;
}


//+------------------------------------------------------------------+
//| Send CONFIRM message back on tick socket                         |
//+------------------------------------------------------------------+
void SendConfirm(string req_id, int ticket, string status)
{
   string msg = StringFormat("CONFIRM|%s|%d|%s", req_id, ticket, status);
   SendTickMsg(msg);
}
