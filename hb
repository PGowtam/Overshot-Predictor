(.venv) (base) gopo@Gowtams-MacBook-Air BrickOfTicks_Trader % clear
(.venv) (base) gopo@Gowtams-MacBook-Air BrickOfTicks_Trader % python -m bridge.main

2026-05-20 02:01:29,842 | INFO | Initializing BridgeEngine components...
2026-05-20 02:01:29,842 | INFO | RenkoBuilder initialized: day_open=1.00000, brick_size=0.00295
2026-05-20 02:01:29,842 | INFO | Loaded existing state from logs/state.json
2026-05-20 02:01:29,842 | INFO | PathOptimizer: Warming up JIT compiler...
2026-05-20 02:01:30,484 | INFO | PathOptimizer: JIT compilation complete.
2026-05-20 02:01:30,484 | INFO | Starting BridgeEngine...
2026-05-20 02:01:30,484 | INFO | Loading Keras model for fold 1 from /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/models/fold_1/model.keras...
2026-05-20 02:01:30,526 | INFO | Loading Keras model for fold 2 from /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/models/fold_2/model.keras...
2026-05-20 02:01:30,553 | INFO | Loading Keras model for fold 3 from /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/models/fold_3/model.keras...
2026-05-20 02:01:30,577 | INFO | Successfully loaded 3 models.
2026-05-20 02:01:30,577 | INFO | TickReceiver listening on 127.0.0.1:9000
2026-05-20 02:01:30,577 | INFO | Waiting for DAYOPEN from MT5 (max 30s)...
2026-05-20 02:01:31,707 | INFO | EA connected from ('127.0.0.1', 61079)
2026-05-20 02:01:31,707 | INFO | Day open price: 4567.07
2026-05-20 02:01:31,712 | INFO | Day open received: 4567.07
2026-05-20 02:01:31,712 | INFO | Waiting for HDONE (history batch complete) from MT5 (max 120s)...
2026-05-20 02:03:19,738 | INFO | History batch complete: 4155992 ticks (received 4155984)
2026-05-20 02:03:19,788 | INFO | Received 4155984 historical ticks from EA.
2026-05-20 02:03:20,529 | INFO | Deduplication: 4155984 → 4154178 ticks (1806 duplicates removed)
2026-05-20 02:03:21,105 | INFO | PathOptimizer: 4154178 ticks, range: 2026-05-12 23:31 → 2026-05-19 23:31 UTC, price: 4714.20 → 4483.52
2026-05-20 02:03:21,166 | INFO | PathOptimizer: Found 6 days: 2026-05-12(5419t), 2026-05-13(660056t), 2026-05-14(712383t), 2026-05-15(1022938t), 2026-05-18(966084t), 2026-05-19(787298t)
2026-05-20 02:03:21,166 | INFO | PathOptimizer: Anchor day 2026-05-12 | Open: 4714.20, High: 4715.77, Low: 4712.19
2026-05-20 02:03:21,167 | INFO | PathOptimizer: Testing 26 candidates with Numba JIT...
2026-05-20 02:03:21,326 | INFO | PathOptimizer: Best anchor = 4714.42 | Historical PnL = -4.50 | Start index = 0 | Candidates tested = 26
2026-05-20 02:03:21,326 | INFO | Path Optimization complete: anchor=4714.42, profit=-4.50, start_idx=0
2026-05-20 02:03:21,326 | INFO | RenkoBuilder initialized: day_open=4714.41510, brick_size=13.90752
2026-05-20 02:03:21,326 | INFO | Brick size updated: 13.90752 → 13.47286
2026-05-20 02:03:21,326 | INFO | Startup correction: current_price snapped to true day_open 4714.41510
2026-05-20 02:03:21,327 | INFO | Replaying ticks: features from idx=0, renko from idx=0...
2026-05-20 02:03:30,379 | INFO | Warmup PASSED: 53 bricks, 1000 ticks tracked.
2026-05-20 02:03:30,382 | INFO | Saved 53 historical bricks to /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/bridge/logs/renko_history_snapshot.csv
2026-05-20 02:03:30,382 | INFO | LAST WARMUP BRICK: Dir=UP, Close=4485.37654, Seq='00010110000000000001000111000001111110011000000011010'
2026-05-20 02:03:30,395 | INFO | CommandSender listening on 127.0.0.1:9001 — waiting for EA...
2026-05-20 02:04:00,396 | ERROR | CommandSender: No EA connection within 30s
2026-05-20 02:04:00,396 | WARNING | Could not connect CommandSender on startup: No EA connection on port 9001 within 30s. Will reconnect later.
2026-05-20 02:04:00,396 | INFO | BridgeEngine is LIVE and waiting for streaming ticks.
2026-05-20 02:05:04,371 | INFO | History batch complete: 4155968 ticks (received 8105377)
zsh: killed     python -m bridge.main
(.venv) (base) gopo@Gowtams-MacBook-Air BrickOfTicks_Trader % python -m bridge.main

2026-05-20 02:17:11,641 | INFO | Initializing BridgeEngine components...
2026-05-20 02:17:11,641 | INFO | RenkoBuilder initialized: day_open=1.00000, brick_size=0.00295
2026-05-20 02:17:11,641 | INFO | Loaded existing state from logs/state.json
2026-05-20 02:17:11,641 | INFO | PathOptimizer: Warming up JIT compiler...
2026-05-20 02:17:12,307 | INFO | PathOptimizer: JIT compilation complete.
2026-05-20 02:17:12,307 | INFO | Starting BridgeEngine...
2026-05-20 02:17:12,307 | INFO | Loading Keras model for fold 1 from /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/models/fold_1/model.keras...
2026-05-20 02:17:12,367 | INFO | Loading Keras model for fold 2 from /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/models/fold_2/model.keras...
2026-05-20 02:17:12,390 | INFO | Loading Keras model for fold 3 from /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/models/fold_3/model.keras...
2026-05-20 02:17:12,413 | INFO | Successfully loaded 3 models.
2026-05-20 02:17:12,414 | INFO | TickReceiver listening on 127.0.0.1:9000
2026-05-20 02:17:12,414 | INFO | CommandSender listening on 127.0.0.1:9001 — waiting for EA asynchronously...
2026-05-20 02:17:12,414 | INFO | Waiting for DAYOPEN from MT5 (max 30s)...
2026-05-20 02:17:13,459 | INFO | EA connected from ('127.0.0.1', 61388)
2026-05-20 02:17:13,459 | INFO | Day open price: 4567.07
2026-05-20 02:17:13,553 | INFO | Day open received: 4567.07
2026-05-20 02:17:13,553 | INFO | Waiting for HDONE (history batch complete) from MT5 (max 120s)...
2026-05-20 02:19:03,106 | INFO | History batch complete: 4155567 ticks (received 4155560)
2026-05-20 02:19:03,257 | INFO | Received 4155560 historical ticks from EA.
2026-05-20 02:19:03,984 | INFO | Deduplication: 4155560 → 4153754 ticks (1806 duplicates removed)
2026-05-20 02:19:04,547 | INFO | PathOptimizer: 4153754 ticks, range: 2026-05-12 23:47 → 2026-05-19 23:47 UTC, price: 4713.89 → 4483.50
2026-05-20 02:19:04,562 | INFO | PathOptimizer: Found 6 days: 2026-05-12(2511t), 2026-05-13(660056t), 2026-05-14(712383t), 2026-05-15(1022938t), 2026-05-18(966084t), 2026-05-19(789782t)
2026-05-20 02:19:04,563 | INFO | PathOptimizer: Anchor day 2026-05-12 | Open: 4713.89, High: 4715.77, Low: 4713.16
2026-05-20 02:19:04,563 | INFO | PathOptimizer: Testing 19 candidates with Numba JIT...
2026-05-20 02:19:04,715 | INFO | PathOptimizer: Best anchor = 4714.41 | Historical PnL = -4.50 | Start index = 0 | Candidates tested = 19
2026-05-20 02:19:04,716 | INFO | Path Optimization complete: anchor=4714.41, profit=-4.50, start_idx=0
2026-05-20 02:19:04,716 | INFO | RenkoBuilder initialized: day_open=4714.41154, brick_size=13.90751
2026-05-20 02:19:04,716 | INFO | Brick size updated: 13.90751 → 13.47286
2026-05-20 02:19:04,716 | INFO | Startup correction: current_price snapped to true day_open 4714.41154
2026-05-20 02:19:04,716 | INFO | Replaying ticks: features from idx=0, renko from idx=0...
2026-05-20 02:19:13,668 | INFO | Warmup PASSED: 53 bricks, 1000 ticks tracked.
2026-05-20 02:19:13,670 | INFO | Saved 53 historical bricks to /Users/gopo/Quant Projects/CAPSTONE/Overshot/BrickOfTicks_Trader/bridge/logs/renko_history_snapshot.csv
2026-05-20 02:19:13,670 | INFO | LAST WARMUP BRICK: Dir=UP, Close=4485.37298, Seq='00010110000000000001000111000001111110011000000011010'
2026-05-20 02:19:13,684 | INFO | BridgeEngine is LIVE and waiting for streaming ticks.
2026-05-20 02:20:53,660 | INFO | History batch complete: 4155367 ticks (received 8302190)
2026-05-20 02:20:53,668 | INFO | CommandSender: EA connected from ('127.0.0.1', 61438)
2026-05-20 02:32:59,047 | CRITICAL | 3 consecutive 60s timeouts! Entering DEGRADED MODE. Halting trading execution.
^C2026-05-20 02:33:45,893 | INFO | KeyboardInterrupt caught. Commencing graceful shutdown...
2026-05-20 02:33:45,939 | INFO | No closed trades to report.
2026-05-20 02:33:45,939 | INFO | State saved. Session summary generated. Exiting.
(.venv) (base) gopo@Gowtams-MacBook-Air BrickOfTicks_Trader % 