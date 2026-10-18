"""Run the reproducible project stages in order."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
for script in ['inventory.py', 'report_inventory.py', 'build_data.py', 'train_models.py', 'evaluate_rul.py', 'review_results.py', 'make_report.py']:
    print(f'Running {script}', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script)], cwd=ROOT, env=os.environ.copy(), check=True)
