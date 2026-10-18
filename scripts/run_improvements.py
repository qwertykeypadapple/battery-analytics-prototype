"""Run one preregistered stage without changing the baseline artifacts."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['E0','E1','E2','E3','E4','E5','E6','E7','E8','E9','report'])
    args = parser.parse_args()
    from battery_analytics.improvements.core import verify_baseline
    verify_baseline()
    if args.stage in ['E0', 'E1']:
        from battery_analytics.improvements import core
        getattr(core, args.stage.lower())()
    else:
        from battery_analytics.improvements import stages
        getattr(stages, args.stage.lower())()
