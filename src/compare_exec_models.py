"""
Compare Execution Models (Baseline vs Markov)
=============================================
Runs the 2026 Live Engine Simulation on both Model A (Baseline) and Model B (Markov).
Generates a side-by-side comparison report.
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def run_simulation(script_name):
    script_path = BASE_DIR / "src" / script_name
    print(f"\n🚀 Running {script_name}...")
    
    # Run the script as a subprocess
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error running {script_name}:\n")
        print(result.stderr)
    else:
        print(f"✅ {script_name} completed successfully.")


def extract_results(report_path):
    if not report_path.exists():
        return None
        
    results = {}
    with open(report_path, "r") as f:
        lines = f.readlines()
        
    in_summary = False
    for line in lines:
        if "## Results Summary" in line:
            in_summary = True
            continue
        if in_summary and "## Daily Breakdown" in line:
            break
            
        if in_summary and "|" in line and "Metric" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                # Remove markdown bolding for clean extraction
                metric = parts[0].replace("**", "")
                val = parts[1].replace("**", "")
                results[metric] = val
                
    return results


def main():
    print("=" * 60)
    print(" Execution Models Live 2026 Comparison")
    print("=" * 60)
    
    # 1. Run Baseline
    run_simulation("evaluate_live_baseline.py")
    
    # 2. Run Markov
    run_simulation("evaluate_live_markov.py")
    
    # 3. Aggregate Reports
    baseline_report = BASE_DIR / "outputs" / "experiments" / "sim_2026_exec_baseline" / "sim_session_report.md"
    markov_report = BASE_DIR / "outputs" / "experiments" / "sim_2026_exec_markov" / "sim_session_report.md"
    
    base_res = extract_results(baseline_report)
    mark_res = extract_results(markov_report)
    
    if not base_res or not mark_res:
        print("❌ Could not extract results from one or both reports.")
        return
        
    comparison_md = f"""# Dual-Model Execution Comparison (2026 Live Simulation)

This report compares Model A (Baseline Execution) against Model B (Markov Sequences) on the 2026 out-of-sample tick data, using strict Bid/Ask live execution spreads at `K=0.00118`.

## Performance Metrics

| Metric | Model A (Baseline) | Model B (Markov) | Difference |
| :--- | :--- | :--- | :--- |
"""
    
    metrics = [
        "Total Resolved Trades", "Wins", "Losses", 
        "Win Rate", "Total PnL", "EV per Trade", "Total Signals"
    ]
    
    for m in metrics:
        vA = base_res.get(m, "N/A")
        vB = mark_res.get(m, "N/A")
        comparison_md += f"| **{m}** | {vA} | {vB} | |\n"
        
    out_path = BASE_DIR / "outputs" / "exec_comparison_report.md"
    with open(out_path, "w") as f:
        f.write(comparison_md)
        
    print(f"\n✅ Comparison report saved to {out_path}")
    print("You can view the detailed breakdown in the markdown file.")

if __name__ == "__main__":
    main()
