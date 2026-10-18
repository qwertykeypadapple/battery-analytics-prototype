"""Refresh main report from saved experiment results; does not fit models."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from battery_analytics.improvements.summary import sync_main_results

if __name__=='__main__':
    if not sync_main_results():
        raise SystemExit('Run the improvement and fleet forecasting stages first.')
