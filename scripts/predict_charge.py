"""Score a new single-cell-equivalent charge trace, without any capacity label."""
import argparse
from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from battery_analytics.common import load_config
from battery_analytics.features import extract_charge_features

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('csv', type=Path, help='Columns: time_s, voltage_v, current_a, optional temperature_c')
parser.add_argument('--profile', required=True, choices=['nasa', 'oxford', 'calce'], help='Configured single-cell voltage window. No Baaz-specific profile has been validated.')
parser.add_argument('--rated-capacity-ah', required=True, type=float)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
if args.rated_capacity_ah <= 0:
    parser.error('Rated capacity must be positive')
data = pd.read_csv(args.csv)
required = ['time_s', 'voltage_v', 'current_a']
if set(required) - set(data):
    parser.error(f'Missing columns: {sorted(set(required) - set(data))}')
config = load_config()
features = extract_charge_features(data.time_s, data.voltage_v, data.current_a,
    data.temperature_c if 'temperature_c' in data else None, args.rated_capacity_ah, config['features'], args.profile)
output = dict(eligible=features['feature_valid'], reason=features['feature_reason'],
    use='Laboratory reference estimate. Not calibrated or validated on Baaz packs. Single-cell-equivalent voltage required.')
if features['feature_valid']:
    artifact = joblib.load(ROOT / 'models/lab_reference_model.joblib')
    x = pd.DataFrame([{k: features[k] for k in artifact['features']}])
    prediction = float(artifact['estimator'].predict(x)[0])
    output.update(soh=prediction, soh_pct=prediction * 100, estimated_capacity_ah=prediction * args.rated_capacity_ah,
        features=x.iloc[0].to_dict(), outside_training_ranges=[k for k, (lo, hi) in artifact['feature_ranges'].items() if not lo <= features[k] <= hi],
        rul='Unavailable from a single charge trace. Requires a causal history of predicted SoH.')
result = json.dumps(output, indent=2)
if args.output:
    args.output.write_text(result + '\n')
print(result)
