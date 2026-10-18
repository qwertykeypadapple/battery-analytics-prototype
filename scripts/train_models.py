import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))
from battery_analytics.train import train
if __name__ == '__main__':
    train()
