import os
import csv
import logging
import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Logs raw DL predictions, tracks their execution, and generates
    quantitative validation reports against the theoretical baseline.
    """
    COLUMNS = [
        "timestamp", "brick_dir", 
        "fold1_pw", "fold1_os", "fold2_pw", "fold2_os", "fold3_pw", "fold3_os", 
        "votes", "action", 
        "entry", "sl", "tp", "ticket", "outcome", "pnl_pts", "entry_spread_pts"
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
        Appends a new row with the neural network outputs.
        decision is the output from EnsemblePredictor.predict()
        """
        row = {col: "" for col in self.COLUMNS}
        row["timestamp"] = brick_timestamp
        row["brick_dir"] = brick_dir
        row["outcome"] = "PENDING"
        
        details = decision.get("details", [])
        if len(details) == 3:
            row["fold1_pw"] = details[0]["prob_win"]
            row["fold1_os"] = details[0]["pred_os"]
            row["fold2_pw"] = details[1]["prob_win"]
            row["fold2_os"] = details[1]["pred_os"]
            row["fold3_pw"] = details[2]["prob_win"]
            row["fold3_os"] = details[2]["pred_os"]
            
        row["votes"] = decision.get("votes", 0)
        row["action"] = decision.get("action", 0)
        
        # Append mode is fine here
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)
        logger.info(f"Logged new signal at TS {brick_timestamp} -> Action: {row['action']}")

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
        Calculates WR and Expectancy to compare against the 90.3% holdout baseline.
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
        
        baseline_wr = 90.3
        drift = win_rate - baseline_wr
        
        total_pnl = df_closed["pnl_pts"].sum()
        expectancy = total_pnl / trades if trades > 0 else 0
        
        report = f"# Session Trade Report\n\n"
        report += f"**Total Closed Trades:** {trades}\n"
        report += f"**Wins:** {wins}\n"
        report += f"**Win Rate:** {win_rate:.2f}% (Baseline: {baseline_wr}%)\n"
        report += f"**Drift from Baseline:** {drift:+.2f}%\n"
        report += f"**Total PnL Points:** {total_pnl:.2f}\n"
        report += f"**Expectancy (Pts/Trade):** {expectancy:.2f}\n\n"
        
        report += "### Actionable Insights\n"
        if drift < -5.0:
            report += "- 🚨 **WARNING**: Win rate has degraded significantly below the 5% tolerance threshold. Consider halting.\n"
        else:
            report += "- ✅ System is operating within acceptable baseline bounds.\n"
            
        with open(report_path, 'w') as f:
            f.write(report)
            
        logger.info(f"Generated session report at {report_path} (WR: {win_rate:.1f}%)")
