"""
MTPATSC Trader — Trade Logger
================================
Logs MTPATSC predictions and trade outcomes.
Tracks setup type (T1-T4) and 5-class probability vector.
"""

import os
import csv
import logging
import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Logs raw MTPATSC predictions, tracks their execution, and generates
    quantitative validation reports.
    """
    COLUMNS = [
        "timestamp", "brick_dir", "setup_type",
        "prob_t0", "prob_t1", "prob_t2", "prob_t3", "prob_t4",
        "action", "reason",
        "entry", "sl", "tp", "rr", "ticket", "outcome", "pnl_pts", "entry_spread_pts"
    ]

    def __init__(self, filepath="logs/trades.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)

        # Initialize CSV if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def _read_all(self):
        rows = []
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        return rows

    def _write_all(self, rows):
        # Write to temp file then replace for atomic integrity
        tmp_file = self.filepath + ".tmp"
        with open(tmp_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_file, self.filepath)

    def log_signal(self, brick_timestamp: int, brick_dir: int, decision: dict):
        """
        Appends a new row with the MTPATSC prediction outputs.
        decision is the output from MTPatscPredictor.predict()
        """
        row = {col: "" for col in self.COLUMNS}
        row["timestamp"] = brick_timestamp
        row["brick_dir"] = brick_dir
        row["outcome"] = "PENDING"

        probs = decision.get("probs", np.zeros(5))
        if len(probs) == 5:
            row["prob_t0"] = f"{probs[0]:.4f}"
            row["prob_t1"] = f"{probs[1]:.4f}"
            row["prob_t2"] = f"{probs[2]:.4f}"
            row["prob_t3"] = f"{probs[3]:.4f}"
            row["prob_t4"] = f"{probs[4]:.4f}"

        row["setup_type"] = f"T{decision.get('setup_type', 0)}"
        row["action"] = decision.get("action", 0)
        row["reason"] = decision.get("reason", "")
        row["rr"] = decision.get("rr", 0)

        # Append mode is fine here
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)
        logger.info(f"Logged signal at TS {brick_timestamp} → Action: {row['action']}, Setup: {row['setup_type']}")

    def log_order(self, ticket: int, entry: float, sl: float, tp: float, direction: int, entry_spread_pts: float = 0.0):
        """
        Updates the most recent PENDING signal with order execution details.
        """
        rows = self._read_all()
        # Find the last PENDING row
        updated = False
        for row in reversed(rows):
            if row["outcome"] == "PENDING" and int(row.get("action", 0)) == 1:
                row["ticket"] = ticket
                row["entry"] = entry
                row["sl"] = sl
                row["tp"] = tp
                row["entry_spread_pts"] = entry_spread_pts
                row["outcome"] = "OPEN"
                updated = True
                break

        if updated:
            self._write_all(rows)
            logger.info(f"Logged order execution for ticket {ticket}")
        else:
            logger.warning(f"Could not find a PENDING signal to map order ticket {ticket} to.")

    def log_outcome(self, ticket: int, outcome: str, pnl_pts: float):
        """
        Updates the specific ticket's row with the final outcome (WIN/LOSS/BE).
        """
        rows = self._read_all()
        updated = False
        for row in reversed(rows):
            if str(row.get("ticket", "")) == str(ticket):
                row["outcome"] = outcome
                row["pnl_pts"] = pnl_pts
                updated = True
                break

        if updated:
            self._write_all(rows)
            logger.info(f"Logged outcome for ticket {ticket}: {outcome} ({pnl_pts} pts)")
        else:
            logger.warning(f"Could not find ticket {ticket} to log outcome.")

    def generate_session_report(self, report_path="logs/session_report.md"):
        """
        Calculates WR and Expectancy per setup type.
        """
        if not os.path.exists(self.filepath):
            logger.warning("No trades.csv found. Cannot generate report.")
            return

        df = pd.read_csv(self.filepath)
        df_closed = df[df["outcome"].isin(["WIN", "LOSS", "BE"])]

        trades = len(df_closed)
        if trades == 0:
            logger.info("No closed trades to report.")
            return

        wins = len(df_closed[df_closed["outcome"] == "WIN"])
        win_rate = (wins / trades) * 100

        total_pnl = df_closed["pnl_pts"].sum()
        expectancy = total_pnl / trades if trades > 0 else 0

        report = f"# MTPATSC Session Trade Report\n\n"
        report += f"**Total Closed Trades:** {trades}\n"
        report += f"**Wins:** {wins}\n"
        report += f"**Win Rate:** {win_rate:.2f}%\n"
        report += f"**Total PnL Points:** {total_pnl:.2f}\n"
        report += f"**Expectancy (Pts/Trade):** {expectancy:.2f}\n\n"

        # Per-setup breakdown
        report += "## Per-Setup Breakdown\n\n"
        report += "| Setup | Trades | Win Rate | Expectancy |\n"
        report += "|-------|--------|----------|------------|\n"

        for st in ["T1", "T2", "T3", "T4"]:
            df_st = df_closed[df_closed["setup_type"] == st]
            if len(df_st) > 0:
                st_wins = len(df_st[df_st["outcome"] == "WIN"])
                st_wr = (st_wins / len(df_st)) * 100
                st_exp = df_st["pnl_pts"].sum() / len(df_st)
                report += f"| {st} | {len(df_st)} | {st_wr:.1f}% | {st_exp:.2f} pts |\n"

        report += "\n"

        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Generated session report at {report_path} (WR: {win_rate:.1f}%)")
