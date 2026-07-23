import os
import time
import pytest
import json
import pandas as pd
from bridge.state import StateManager
from bridge.risk import RiskManager
from bridge.trade_logger import TradeLogger

TEST_STATE_PATH = "logs/test_state.json"
TEST_LOG_PATH = "logs/test_trades.csv"

@pytest.fixture(autouse=True)
def run_around_tests():
    # Cleanup before test
    if os.path.exists(TEST_STATE_PATH): os.remove(TEST_STATE_PATH)
    if os.path.exists(TEST_STATE_PATH + ".tmp"): os.remove(TEST_STATE_PATH + ".tmp")
    if os.path.exists(TEST_LOG_PATH): os.remove(TEST_LOG_PATH)
    if os.path.exists(TEST_LOG_PATH + ".tmp"): os.remove(TEST_LOG_PATH + ".tmp")
    yield
    # Cleanup after test
    if os.path.exists(TEST_STATE_PATH): os.remove(TEST_STATE_PATH)
    if os.path.exists(TEST_LOG_PATH): os.remove(TEST_LOG_PATH)

def test_state_manager_recovery():
    sm1 = StateManager(TEST_STATE_PATH)
    sm1.load()
    sm1.update("active_ticket", 123456)
    sm1.update("daily_pnl", 50.5)
    
    # Simulate process kill and reload
    sm2 = StateManager(TEST_STATE_PATH)
    sm2.load()
    
    assert sm2.get("active_ticket") == 123456
    assert sm2.get("daily_pnl") == 50.5
    assert sm2.get("schema_version") == 2 # Default recovered

def test_atomic_save_truncation_protection():
    sm = StateManager(TEST_STATE_PATH)
    sm.load()
    sm.update("active_ticket", 999)
    
    # Simulate a crash during atomic save
    # We will manually create a corrupted .tmp file, but the StateManager won't try to load it.
    with open(TEST_STATE_PATH + ".tmp", "w") as f:
        f.write("{corrupt_json: ")
        
    # Reload should read the safe original file
    sm2 = StateManager(TEST_STATE_PATH)
    sm2.load()
    assert sm2.get("active_ticket") == 999

def test_risk_daily_limit():
    bs = 7.08
    # 5 * 7.08 = 35.4. Limit is -35.4
    assert RiskManager.check_daily_limit(-40.0, bs) is False
    assert RiskManager.check_daily_limit(-35.0, bs) is True
    assert RiskManager.check_daily_limit(10.0, bs) is True

def test_risk_position_open():
    state1 = {"active_ticket": 123}
    state2 = {"active_ticket": 0}
    
    assert RiskManager.check_position_open(state1) is False
    assert RiskManager.check_position_open(state2) is True

def test_risk_be_trigger():
    bs = 7.08
    entry = 2400.0
    # trigger = 2400 + (0.3125 * 7.08) = 2402.2125
    state = {
        "active_entry": entry,
        "active_brick_size": bs,
        "active_direction": 1
    }
    
    # bid=2402.22 >= 2402.2125
    tick1 = {"bid": 2402.22, "ask": 2402.50}
    assert RiskManager.check_be_trigger(tick1, state) is True
    
    # bid=2402.20 < 2402.2125
    tick2 = {"bid": 2402.20, "ask": 2402.50}
    assert RiskManager.check_be_trigger(tick2, state) is False

def test_trade_logger_pipeline():
    logger = TradeLogger(TEST_LOG_PATH)
    
    decision = {
        "action": 1,
        "votes": 2,
        "details": [
            {"prob_win": 0.6, "pred_os": 1.5},
            {"prob_win": 0.4, "pred_os": 1.2},
            {"prob_win": 0.7, "pred_os": 1.6}
        ]
    }
    
    # Log 10 signals
    for i in range(10):
        logger.log_signal(1000 + i, 1, decision)
        
    df = pd.read_csv(TEST_LOG_PATH)
    assert len(df) == 10
    
    # Test log_order maps to the latest PENDING row
    logger.log_order(9999, 2400.0, 2390.0, 2420.0, 1, 0.5)
    
    df = pd.read_csv(TEST_LOG_PATH)
    last_row = df.iloc[-1]
    assert last_row["ticket"] == 9999
    assert last_row["outcome"] == "OPEN"
    assert last_row["entry_spread_pts"] == 0.5
    
    # Test log_outcome
    logger.log_outcome(9999, "WIN", 15.5)
    
    df = pd.read_csv(TEST_LOG_PATH)
    last_row = df.iloc[-1]
    assert last_row["outcome"] == "WIN"
    assert last_row["pnl_pts"] == 15.5
    
    # Test Report generation
    report_path = "logs/test_report.md"
    logger.generate_session_report(report_path)
    assert os.path.exists(report_path)
    
    with open(report_path, "r") as f:
        content = f.read()
        assert "Wins:" in content
        assert "15.5" in content
        
    if os.path.exists(report_path):
        os.remove(report_path)
