# Battery analytics laboratory prototype

A charge-only SoH and RUL prototype using the already-downloaded NASA PCoE, Oxford and CALCE Kaggle mirrors. It includes an inventory, dataset-specific loaders, a cached unified table, explainable models, grouped validation, a Streamlit dashboard and a skeptical review.

**Current finding:** the training-selected workflow achieves 3.72 SoH percentage points cell-macro MAE on 41 unseen lab cells, but 13.89 points when an entire dataset is unseen. RUL and first replacement-alert timing are weak. This is a prototype for a calibration pilot, with no Baaz data, pack-level validation or measured operational savings.

## Start the dashboard in this workspace

```bash
cd /Users/aman/Documents/iitd/battery_analytics
.venv/bin/python -m streamlit run app.py
```

Open http://127.0.0.1:8501. The cached results are already built. The environment in this workspace uses the available scientific Python packages; it does not need a download to run.

## Share the local dashboard with ngrok

Keep Streamlit running, then open a second terminal:

```bash
ngrok http http://127.0.0.1:8501
```

Use the HTTPS forwarding URL printed by ngrok. Visitors on the free plan may need to click **Visit Site** once. The public link forwards to this computer and works while the computer, Streamlit and ngrok remain running. Ngrok authentication is configured outside this repository.

## Repository contents

The repository includes source code, tests, the fitted laboratory reference model, processed per-cycle tables, held-out predictions and reports. The saved artifacts let the dashboard run without reparsing or retraining. Original dataset downloads and raw traces remain outside the project; reproducing the parsing and training stages requires those downloads at the configured paths. Local environments, caches and credentials are ignored.

## Fresh environment

Python 3.11 is recommended for the pinned PyPI package versions. The current workspace was also tested with its available Python 3.12 scientific environment. Recreate the virtual environment when moving the project to another computer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Raw dataset downloads and original archives are not included in this project. Configure `data_root` in `config.yaml` if the downloaded data move. Expected paths relative to the project are:

```text
../datasets/nasa_battery/extracted/cleaned_dataset/metadata.csv
../datasets/nasa_battery/extracted/cleaned_dataset/data/*.csv
../datasets/nasa_battery/extracted/cleaned_dataset/extra_infos/*.txt
../datasets/oxford_battery/extracted/Oxford_Battery_Degradation_Dataset_1.mat
../datasets/calce_battery/extracted/*timeseries.csv
../datasets/calce_battery/extracted/*cycle_data.csv
```

## Reproduce everything

```bash
python scripts/run_all.py
python -m pytest -q
python -m streamlit run app.py
```

Or run stages separately:

```bash
python scripts/inventory.py
python scripts/report_inventory.py
python scripts/build_data.py
python scripts/train_models.py
python scripts/evaluate_rul.py
python scripts/review_results.py
python scripts/make_report.py
```

The inventory runs before model code. Parsing is cached using source file signatures, configuration and loader/feature/QC code. `python scripts/build_data.py --force` explicitly rebuilds the slow parse. Re-run training and subsequent evaluation after changing features, QC, model specifications or data. Re-run RUL/review/report stages after changing forecast or alert rules. The dashboard does not train models.

The final `models/lab_reference_model.joblib` is a refit for a future external battery with the same feature specification. Never use its fitted predictions on these training cells as validation. Dashboard curves use only held-out predictions.

To score a new charge trace with columns `time_s`, `voltage_v`, `current_a` and optional `temperature_c`:

```bash
python scripts/predict_charge.py path/to/charge.csv --profile calce --rated-capacity-ah 1.35
```

This produces an explicit eligible/skip result and a laboratory reference estimate. The profile chooses a configured single-cell voltage window; it does not establish calibration on a different cell or pack. RUL needs a history, rather than one charge trace.

## Layout

```text
app.py                         Streamlit fleet and validation views
config.yaml                    Data paths, voltage windows, QC, forecast and alert settings
src/battery_analytics/
  loaders.py                   NASA CSV, Oxford MATLAB, CALCE episode parsers
  features.py                  Isolated charge-only feature extraction and allowlists
  build.py                     Cached common table, references and exclusion logs
  models.py                    Fixed candidate models and training-only selection
  train.py                     Whole-cell and whole-dataset validation
  evaluation.py                Percentage-point errors and trailing-only smoothing
  rul.py                       Linear/exponential forecasts and censoring-aware backtests
  fleet.py                     As-of fleet snapshot and explicit unsupported states
data/processed/                Parquet and CSV caches, reasons and fingerprints
models/                        Laboratory reference estimator and model card
reports/inventory/             Full file schemas, documentation and per-cell inventory
reports/evaluation/            Held-out predictions, metrics and split audit
reports/rul/                   Reference endpoints, forecasts and censoring-aware metrics
reports/review/                Low-SoH results, life-stage errors, alert and feature audits
reports/figures/               Standalone scientific plots
reports/results_summary.md     Full results with limitations and Baaz integration plan
reports/skeptical_review.md    Written pitch review and complete evidence appendix
tests/                         Scientific leakage, units, chronology and app checks
```

## Definitions and methodological choices

- SoH is **independently measured discharge Ah / documented rated Ah**. It is not divided by each cell's first or maximum observed capacity. Nominal ratings are NASA 2.0Ah, Oxford 0.740Ah and CALCE CX2 1.350Ah. Available capacity still depends on the test conditions.
- Common predictors are three voltage-interval charging times, their sum and charging C-rate. IDs, source age/cycle number, absolute timestamps, full-charge capacity and all discharge channels are excluded from the feature allowlist.
- Oxford's charging current is reconstructed from the charging-only `q` derivative. Its independent C1 discharge counter supplies the label. Observed serial-day timestamps are multiplied by 86,400. Its C1 traces do not include a CV taper.
- NASA pairs to the next discharge. CALCE uses charge/discharge order, with measured discharge-counter excursions checked against negative-current integration. Oxford pairs to a nearby same-checkpoint discharge reference and records that it precedes the charge.
- Arrivals already above the lower voltage are excluded. Interrupted or unresolved upper windows, physically inconsistent labels and incomplete nominal CC–CV charges remain in logs. These exclusions do not guarantee all remaining labels are perfect.
- All candidate hyperparameters are fixed. Separate whole training cells select the model family inside each outer fold. No test cell enters fitting, scaling, inner selection or row-level early stopping. Leave-one-dataset-out trains on the other two datasets.
- Report MAE/RMSE in percentage points, both raw and trailing-five-observation median. Cell-macro averages are the headline, while row-weighted and equal-dataset aggregates remain available. Repeated measurements are not independent batteries.
- Temperature and CV are explicit feature-complete variants. Missing CALCE temperature and Oxford CV information are never filled with zero.
- RUL uses only the prediction history available at today. The first observed 70% reference crossing is the endpoint. Cells already below threshold are left-censored; cells never crossing are right-censored, without invented exact RUL errors.
- Days are an editable scenario: `cycles / (swaps_per_day × test_cycles_per_swap)`, default two swaps/day and one source lab cycle/swap. Actual partial swaps need a measured throughput conversion.
- Default alert backtesting scores the **first alarm**, not whether a model eventually emits any warning. At a 14-day horizon, the current model has zero timely first alerts among 19 observed lab threshold events. This is a material failure, not a measured field-failure catch rate.

## Dashboard use

The fleet view starts with CALCE and a cycle within all seven cells' recorded histories. Change datasets, today, raw/median predictions, linear/exponential RUL, swap assumptions and alert thresholds in the controls. A historical record ending before today has an explicit state. Blank RUL means insufficient or unsupported history, never an invented healthy lifetime.

The Validation tab always exposes results for all datasets, independent of which fleet is selected. Future recorded data are hidden by default on battery plots; the retrospective checkbox affects plotting only. Full data and source limitations are under Data and methods.

## Evidence and sources

The original Kaggle ZIP files, source links, licence metadata and download checksums remain in `../datasets/*_battery/`. Primary protocol documentation is linked in the inventory and results summary. Several Kaggle mirrors do not establish complete licensing/provenance; inspect their source terms for any intended redistribution or commercial use.

This project contains no federated-learning implementation and no communication with Baaz. It provides no evidence yet for pack accuracy, prevented stranding, life extension, energy savings or operation above 45°C in Delhi.

## Preregistered improvement experiments and fleet forecasting

The starting state is tagged `baseline-v1`. Baseline artifacts and SHA-256 hashes
are frozen under `reports/improvements/baseline/`; the original model modules
remain unchanged. Read [the preregistration](reports/improvements/preregistration.md)
before [the E0–E9 findings](reports/improvements/results.md).
The original charge-only workflow beats the fixed cycle-count curve (3.72 vs
8.98 pp LOCO), but the added pipelines do not justify replacing the baseline.
Some apparent gains disappear on matched cohorts. Tracker intervals are badly
miscalibrated and target calibration uses full labelled cell trajectories.

Run stages separately, in order, from the project root:

```bash
python scripts/run_improvements.py E0
python scripts/run_improvements.py E1
python scripts/run_improvements.py E2
python scripts/run_improvements.py E3
python scripts/run_improvements.py E4
python scripts/run_improvements.py E5
python scripts/run_improvements.py E6
python scripts/run_improvements.py E7
python scripts/run_improvements.py E8
python scripts/run_improvements.py E9
python scripts/run_improvements.py report
python scripts/fleet_forecast.py
python scripts/update_results_summary.py
python -m pytest -q
```

The main `reports/results_summary.md` and `reports/skeptical_review.md` include
the latest experiment and fleet-count findings above a clearly labelled frozen
baseline appendix. Report-generation commands refresh these summaries when all
new artifacts exist. `update_results_summary.py` refreshes only the written
summaries, without fitting models or changing baseline metrics.

Use `.venv/bin/python` in this existing workspace. The frozen artifacts must
match the manifest; rebuild baseline experiments in a separate checkout when
changing their code or data. E2/E8 need the original raw downloads. Other new
stages use saved tables and their prerequisite improvement artifacts. Regenerating
reports does not train models. Every E0–E9 result and all candidate selection
audits remain saved, including negative results and missing prediction coverage.

The dashboard adds model views for the baseline (recommended), the E8
inner-selected known-history candidate, and the E5/E6 probabilistic tracker.
The tracker's uncertainty bands, probability alerts and E7 flag are explicitly
experimental; its uncertainty is never assigned to the E8 model. The
**Improvement experiments** and **Fleet budgeting** tabs expose the results.

Fleet count forecasting has its own [preregistration](reports/fleet_forecast/preregistration.md)
and [results](reports/fleet_forecast/results.md). It uses held-out point-RUL
counts at 30/60/90 scenario days, comparing cycle-only and historical-rate
baselines. Complete-follow-up matched-count MAE is 0.37 batteries vs 0.80 and
0.43, but coverage is sparse and Oxford contributes zero-incident windows.
Full-fleet outcome/prediction bounds and bias are reported. This is not validated
replacement budgeting. No ngrok token or other credentials are stored here.
