//+------------------------------------------------------------------+
//|                                                  TickExporter.mq5 |
//|                                  Copyright 2026, BrickOfTicks     |
//|                                             https://localhost     |
//+------------------------------------------------------------------+
//| Single-purpose EA: Export historical tick data over TCP socket.    |
//|                                                                   |
//| Architecture:                                                     |
//|   Connects as CLIENT to Python tick_collector.py on port 9100.    |
//|   Streams all XAUUSD ticks for a date range using CopyTicksRange. |
//|   Chunks by calendar week to avoid MT5 memory limits.             |
//|                                                                   |
//| Protocol: UTF-8, newline-terminated, pipe-delimited               |
//|   HTICK|<time_msc>|<bid>|<ask>|<bid_vol>|<ask_vol>               |
//|   HDONE|<total_count>                                             |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, BrickOfTicks"
#property link      "https://localhost"
#property version   "1.00"
#property strict

//--- Input parameters
input int    InpPort           = 9100;    // Python collector port
input int    InpConnectRetries = 10;      // Max connection retry attempts
input int    InpConnectWaitMs  = 2000;    // Wait between retries (ms)
input string InpStartDate      = "2026.01.01";  // Export start date (YYYY.MM.DD)
input string InpEndDate        = "2026.05.31";  // Export end date (YYYY.MM.DD)
input int    InpChunkDays      = 7;       // Days per CopyTicksRange chunk

//--- Socket handle
int export_socket = INVALID_HANDLE;
bool connected = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("═══════════════════════════════════════════════════");
   Print("  TickExporter EA v1.00 — Historical Tick Export");
   Print("  Target: ", _Symbol);
   Print("  Range:  ", InpStartDate, " → ", InpEndDate);
   Print("  Port:   ", InpPort);
   Print("═══════════════════════════════════════════════════");

   // ─── Connect to Python collector ────────────────────────────
   export_socket = SocketCreate();
   if(export_socket == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create socket. Error: ", GetLastError());
      return INIT_FAILED;
   }

   connected = false;
   for(int i = 0; i < InpConnectRetries; i++)
   {
      if(SocketConnect(export_socket, "127.0.0.1", InpPort, InpConnectWaitMs))
      {
         connected = true;
         Print("✓ Connected to Python collector on 127.0.0.1:", InpPort,
               " (attempt ", i+1, "/", InpConnectRetries, ")");
         break;
      }
      Print("  Retry ", i+1, "/", InpConnectRetries,
            " — waiting ", InpConnectWaitMs, "ms...");
      Sleep(InpConnectWaitMs);
   }

   if(!connected)
   {
      Print("CRITICAL: Failed to connect to Python collector.");
      Print("  → Is tick_collector.py running? (python src/tick_collector.py)");
      return INIT_FAILED;
   }

   // ─── Start export on a timer (give OnInit time to return) ────
   EventSetMillisecondTimer(500);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Timer function — triggers the export once                        |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Kill the timer immediately — we only need one trigger
   EventKillTimer();

   if(!connected)
   {
      Print("ERROR: Not connected. Cannot export.");
      return;
   }

   // ─── Parse date range ────────────────────────────────────────
   datetime start_dt = StringToTime(InpStartDate);
   datetime end_dt   = StringToTime(InpEndDate) + 86400; // Include end date fully

   if(start_dt == 0 || end_dt == 0)
   {
      Print("ERROR: Invalid date format. Use YYYY.MM.DD");
      return;
   }

   Print("Starting tick export: ", TimeToString(start_dt, TIME_DATE),
         " → ", TimeToString(end_dt - 1, TIME_DATE));

   // ─── Chunked export ──────────────────────────────────────────
   int total_sent = 0;
   int chunk_num = 0;
   datetime chunk_start = start_dt;

   while(chunk_start < end_dt)
   {
      datetime chunk_end = chunk_start + InpChunkDays * 86400;
      if(chunk_end > end_dt) chunk_end = end_dt;

      chunk_num++;
      Print("  Chunk ", chunk_num, ": ",
            TimeToString(chunk_start, TIME_DATE), " → ",
            TimeToString(chunk_end - 1, TIME_DATE));

      MqlTick ticks[];
      int copied = CopyTicksRange(_Symbol, ticks, COPY_TICKS_ALL,
                                   (ulong)chunk_start * 1000,
                                   (ulong)chunk_end * 1000);

      if(copied <= 0)
      {
         Print("    No ticks in this chunk (", copied, "). Skipping.");
         chunk_start = chunk_end;
         continue;
      }

      Print("    Fetched ", copied, " ticks. Streaming...");

      for(int i = 0; i < copied; i++)
      {
         string msg = StringFormat("HTICK|%lld|%.5f|%.5f|%.2f|%.2f\n",
            ticks[i].time_msc, ticks[i].bid, ticks[i].ask,
            ticks[i].volume_real, ticks[i].volume_real);

         if(!SendMsg(msg))
         {
            Print("CRITICAL: Socket send failed at tick ", total_sent + i);
            Print("  Aborting export.");
            SendMsg(StringFormat("HDONE|%d\n", total_sent));
            return;
         }

         // Yield every 10000 ticks to prevent socket buffer overflow
         if(i > 0 && i % 10000 == 0)
            Sleep(10);
      }

      total_sent += copied;
      Print("    Chunk complete. Running total: ", total_sent, " ticks.");

      // Small sleep between chunks to let Python process
      Sleep(200);

      chunk_start = chunk_end;
   }

   // ─── Send completion marker ──────────────────────────────────
   Sleep(500); // Allow buffer flush
   SendMsg(StringFormat("HDONE|%d\n", total_sent));

   Print("═══════════════════════════════════════════════════");
   Print("  EXPORT COMPLETE: ", total_sent, " total ticks sent");
   Print("  Chunks processed: ", chunk_num);
   Print("═══════════════════════════════════════════════════");
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("TickExporter shutting down. Reason: ", reason);
   if(export_socket != INVALID_HANDLE)
   {
      SocketClose(export_socket);
      export_socket = INVALID_HANDLE;
   }
   Print("TickExporter shutdown complete.");
}

//+------------------------------------------------------------------+
//| Send a message on the export socket                              |
//+------------------------------------------------------------------+
bool SendMsg(string msg)
{
   if(export_socket == INVALID_HANDLE || !connected)
      return false;

   uchar data[];
   int len = StringToCharArray(msg, data, 0, -1, CP_UTF8);
   if(len > 0) len--;  // Strip null terminator

   int bytes_sent = SocketSend(export_socket, data, len);

   if(bytes_sent <= 0)
   {
      int err = GetLastError();
      Print("ERROR: SocketSend failed. Error: ", err);
      connected = false;
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Tick function — not used (export only)                           |
//+------------------------------------------------------------------+
void OnTick()
{
   // No-op: This EA only exports historical data.
}
