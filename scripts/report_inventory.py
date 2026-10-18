"""Format the measured inventory and source documentation before modelling."""
from pathlib import Path
from collections import Counter
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
I = ROOT / "reports/inventory"
cells = pd.read_csv(I / "cells.csv")
summary = json.loads((I / "summary.json").read_text())
anomalies = json.loads((I / "anomalies.json").read_text())
anomalies = [a for a in anomalies if a['issue'] != 'discharge_capacity_above_150pct_nominal']
for p in (ROOT.parent / "datasets/calce_battery/extracted").glob("*cycle_data.csv"):
    d = pd.read_csv(p)
    for _, r in d[d["Discharge_Capacity (Ah)"] > 1.5 * 1.35].iterrows():
        anomalies.append({"dataset": "calce", "source": p.name, "cycle_index": r.Cycle_Index,
                          "issue": "discharge_capacity_above_150pct_nominal", "value_ah": r["Discharge_Capacity (Ah)"]})
(I / "anomalies.json").write_text(json.dumps(anomalies, indent=2) + "\n")
summary['anomaly_counts'] = dict(Counter(a['dataset'] + ':' + a['issue'] for a in anomalies))
(I / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

text = f"""# Data inventory

This inventory was generated before any model code. All {summary['files_scanned']} files under the three downloaded dataset directories were opened according to format. No unreadable file was found. The source data remain unchanged. A readable file can still contain unusable measurements.

| Dataset | Supplied data formats and structure | Cells | Charge records / capacity measurements | Rated capacity | Temperature |
|---|---|---:|---:|---|---|
| NASA PCoE | 7,566 CSV files including one metadata table, 9 readmes; 7,565 individual charge/discharge/EIS records | 34 | 2,815 charge operations / 2,769 numeric capacity labels in 2,794 discharge operations | 2.0 Ah | ambient 4, 22, 24, 43, 44°C; measured cell temperature available |
| Oxford | One MATLAB v5-style readable `.mat`, eight `CellN` structs, `cycNNNN` checkpoints, four trace types per checkpoint | 8 | 519 C1 charge traces / 519 C1 discharge capacities | 0.740 Ah | 40°C chamber documented, measured cell temperature available |
| CALCE | Seven pairs of CSV files: one time series and one cycle summary for each cell | 7 | 12,477 cycle groups contain charge current / 12,480 cycle summaries, including invalid capacities | 1.350 Ah, CX2 family specification | 25°C comes from mirror filenames, both temperature columns are entirely empty |

Counts above describe the downloaded files before any filtering or event segmentation. A vendor cycle index is not necessarily one charge/discharge pair. Capacity measurements with zero, implausible values or incomplete traces are not automatically valid labels.

## Signals, structures and units

NASA metadata columns: `type`, `start_time` (MATLAB date-vector string), `ambient_temperature` (°C), `battery_id`, `test_id` (operation index, not ageing-cycle count), `uid`, `filename`, `Capacity` (Ah), `Re`, `Rct` (ohms). Charge CSVs contain `Voltage_measured` (V), `Current_measured` (A), `Temperature_measured` (°C), `Current_charge` (A), `Voltage_charge` (V), `Time` (s). Discharge CSVs use `Current_load` and `Voltage_load`, contrary to some supplied readme field descriptions. EIS contains complex-valued `Sense_current`, `Battery_current`, `Current_ratio`, `Battery_impedance`, `Rectified_Impedance`. EIS is not a charging trace and will not enter the charge-only feature model. The downloaded NASA version is the cleaned CSV derivative, not the original MATLAB archive.

Oxford observed keys: `Cell1`–`Cell8`, then checkpoint labels such as `cyc0000`, `cyc0100`, then `C1ch`, `C1dc`, `OCVch`, `OCVdc`, each with `t`, `v`, `q`, `T`. There are no Excel sheets. Voltage is V, charge is mAh, temperature is °C. The observed `t` values are around 735,954 and consecutive samples differ by approximately 1/86,400: these are MATLAB serial days and must be converted to seconds, rather than interpreted as seconds from memory. There is no stored current channel. Charging current can be reconstructed locally as `3600 × d(q/1000)/d(t×86400)`. Discharge charge-counter magnitude supplies only the capacity label, never an input feature. The C1 charging traces have essentially constant 1C current up to 4.2V in the inspected traces, so a CV duration or current taper cannot be invented for this dataset. Only periodic characterisation data are present, not every intervening driving cycle.

CALCE time-series columns: `Date_Time`, `Test_Time (s)`, `Cycle_Index`, `Current (A)`, `Voltage (V)`, `Charge_Capacity (Ah)`, `Discharge_Capacity (Ah)`, `Charge_Energy (Wh)`, `Discharge_Energy (Wh)`, `Environment_Temperature (C)`, `Cell_Temperature (C)`. Cycle summaries contain `Cycle_Index`, `Start_Time`, `End_Time`, `Test_Time (s)`, min/max current and voltage, charge/discharge capacity and energy. Start/end summary timestamps are empty. Positive current denotes charging. Temperature column names do not mean measured temperature is available: both are 100% missing in all seven time-series files. The full per-file column lists, missing counts, numeric ranges and lengths are in `inventory/file_manifest.json` and `inventory/schemas.json`.

## Cell chemistry and charging protocols

| Dataset | Chemistry and provenance | Charging protocol | Discharge / capacity caveat |
|---|---|---|---|
| NASA | Commercial Li-ion 18650. Exact cathode is not established by the supplied readmes | 1.5A CC to 4.2V, CV to 20mA, documented consistently in the nine local group readmes | Different load currents and cutoffs, some below the OEM 2.7V limit. Cold available capacity is not automatically irreversible ageing |
| Oxford | Graphite negative electrode, lithium cobalt oxide/lithium nickel cobalt oxide positive electrode, per the authors' experimental paper | Ageing recharge 2C. Recorded characterisation `C1ch` is ~0.74A at 1C to ~4.2V. A CV cutoff current is not established for these stored C1 traces | C1 discharge capacity in Ah from `q`. Characterisation every ~100 ageing cycles, with gaps and earlier cell termination |
| CALCE CX2 | LiCoO2 cathode with trace Mn, 1.35Ah family specification from University of Maryland | Nominal 0.5C = 0.675A CC to 4.2V, CV to <0.05A. Some sessions have rests between CC and CV | Some cell filenames label discharge as 0.5C even when observed current is ~1C. Use measured data and retain this discrepancy |

NASA source: https://data.nasa.gov/dataset/li-ion-battery-aging-datasets and the nine copied group readmes in `inventory/`.

Oxford primary experimental paper: https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/03/Gaussian-Process-Regression-for-In-Situ-Capacity-Estimation-of-Lithium-Ion-Batteries.pdf (Richardson, Birkl, Osborne and Howey, DOI 10.1109/TII.2018.2794997). The original ORA record linked by the mirror returned HTTP 403 during verification. The primary paper and observed traces establish the parameters above. No unsupported CV cutoff is assumed.

CALCE source: https://calce.umd.edu/battery-data and https://web.calce.umd.edu/batteries/data/. `CX2-25` is the Kaggle mirror's cell identifier and is not independently matched to an item in the original protocol list. Its 1.35Ah nominal capacity is a documented CX2-family assignment, not a measured initial capacity. This identity uncertainty is retained.

## Anomalies and missing coverage

- NASA has 25 discharge CSVs with nonfinite measured voltage/current/temperature. Of 2,794 discharge records, 25 capacity fields are nonnumeric/missing and 19 more are zero. There are 187 positive values below 10% of nominal. The cold-test readmes explicitly mention unexplained very low capacities, and the B0049–B0052 series ended following test-control software failure. B0052 has only four numeric capacities among 25 discharge operations. Report such observations and apply trace-based eligibility checks rather than smoothing them away.
- NASA nominal-relative capacity can exceed 100% and cold cells can start below 70%. Such observations will not be rescaled to make every cell start at 100%. Interpretation as test-condition available capacity is necessary.
- CALCE has 65 zero/nonpositive discharge summaries, four further positive values below 10% of nominal, two cycle groups shorter than ten rows, and at least one impossible-looking 12.497Ah capacity for a 1.35Ah cell. These require trace-level checks. No decimal point will be silently repaired.
- The first CALCE cycle inspected contains multiple charge/discharge episodes. Pair by time order and episode boundaries, not merely the shared cycle index or the maximum capacity in its summary row.
- No unreadable Oxford trace, inconsistent vector length or non-increasing time was found in the full nested-curve inventory. Several cells stop around 5,000 cycles and others around 8,100–8,200. Most do not cross 70% in the recorded C1 discharge labels and will be right-censored for RUL.
- NASA has controlled 43–44°C tests and Oxford has 40°C tests. The supplied data do not represent Delhi station charging above 45°C, pack thermal gradients, rider-driven partial swaps or uncontrolled outdoor exposure.

## Per-cell counts before cleaning

"""
cols = ["dataset", "cell_id", "charge_records", "capacity_measurements", "rated_capacity_ah", "ambient_temp_c", "cycle_first", "cycle_last", "capacity_min_ah", "capacity_max_ah"]
text += cells[cols].to_markdown(index=False, floatfmt=".4g") + "\n"
(ROOT / "reports/inventory_report.md").write_text(text)
print(text.split("## Signals")[0])
print("Full report:", ROOT / "reports/inventory_report.md")
