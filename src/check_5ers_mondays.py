import pandas as pd
from pathlib import Path
import logging
import time

BASE_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def main():
    t0 = time.time()
    parquet_path = BASE_DIR / "Data" / "xauusd_ticks_5ers_2026.parquet"
    
    logger.info(f"Loading raw ticks from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Create datetime from time_msc
    df['dt'] = pd.to_datetime(df['time_msc'], unit='ms')
    df['day_name'] = df['dt'].dt.day_name()
    df['date'] = df['dt'].dt.date
    
    # Filter for Mondays
    mondays = df[df['day_name'] == 'Monday']
    
    if len(mondays) == 0:
        logger.info("No Mondays found in the raw ticks!")
        return
        
    logger.info("\n--- STARTING TICK FOR EACH MONDAY (5ERS) ---")
    
    # Group by date and get the minimum timestamp
    for date, group in mondays.groupby('date'):
        first_tick = group['dt'].min()
        logger.info(f"Date: {date} | First Tick: {first_tick}")
        
    logger.info(f"\nTotal script time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
