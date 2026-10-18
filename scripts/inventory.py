"""Read-only inventory. This script deliberately contains no model code."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "datasets"
OUT = PROJECT / "reports" / "inventory"
OUT.mkdir(parents=True, exist_ok=True)


def plain(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=plain, allow_nan=True) + "\n")


def finite_range(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return [float(a.min()), float(a.max())] if len(a) else [None, None]


def main():
    manifest, schema_groups, anomalies, cells, curves = [], {}, [], [], []
    roots = {k: DATA / f"{k}_battery" for k in ("nasa", "oxford", "calce")}
    for name, root in roots.items():
        if not root.exists():
            raise FileNotFoundError(root)
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            entry = {"dataset": name, "path": str(p.relative_to(DATA)), "bytes": p.stat().st_size,
                     "format": p.suffix.lower(), "status": "readable"}
            try:
                if p.suffix == ".zip":
                    with ZipFile(p) as z:
                        entry["archive_members"] = len(z.infolist())
                elif p.suffix == ".json":
                    entry["keys"] = list(json.loads(p.read_text()))
                elif p.suffix == ".txt":
                    text = p.read_text(errors="strict")
                    entry["text_characters"] = len(text)
                    if p.name.startswith("README_"):
                        (OUT / p.name).write_text(text)
                elif p.suffix == ".csv":
                    d = pd.read_csv(p)
                    entry.update(rows=len(d), columns=list(d), dtypes={c: str(t) for c, t in d.dtypes.items()},
                                 missing_by_column={c: int(v) for c, v in d.isna().sum().items()})
                    signature = name + ":" + ",".join(d.columns)
                    schema_groups.setdefault(signature, {"dataset": name, "columns": list(d), "files": 0, "rows": 0})
                    schema_groups[signature]["files"] += 1
                    schema_groups[signature]["rows"] += len(d)
                    if not len(d):
                        anomalies.append({"dataset": name, "source": entry["path"], "issue": "empty_csv"})
                    if name == "nasa" and p.parent.name == "data":
                        for col in ["Time", "Voltage_measured", "Current_measured", "Temperature_measured"]:
                            if col not in d:
                                continue
                            a = pd.to_numeric(d[col], errors="coerce").to_numpy()
                            entry.setdefault("numeric_ranges", {})[col] = finite_range(a)
                            nbad = int((~np.isfinite(a)).sum())
                            if nbad:
                                anomalies.append({"dataset": name, "source": p.name, "issue": f"nonfinite_{col}", "count": nbad})
                            if col == "Time":
                                entry["time_nonpositive_steps"] = int((np.diff(a) <= 0).sum())
                                if (np.diff(a) < 0).any():
                                    anomalies.append({"dataset": name, "source": p.name, "issue": "time_runs_backwards"})
                    elif name == "calce":
                        entry["numeric_ranges"] = {c: finite_range(d[c]) for c in d.select_dtypes("number")}
                        if "timeseries" in p.name:
                            cid = re.search(r"CALCE_(CX2-\d+)_", p.name).group(1)
                            grouped = d.groupby("Cycle_Index", sort=False)
                            charge_count = 0
                            for cyc, g in grouped:
                                if (g["Current (A)"] > .03).any():
                                    charge_count += 1
                                dt = np.diff(g["Test_Time (s)"].to_numpy())
                                if (dt < 0).any():
                                    anomalies.append({"dataset": name, "cell_id": cid, "cycle_index": cyc, "issue": "time_runs_backwards"})
                                if len(g) < 10:
                                    anomalies.append({"dataset": name, "cell_id": cid, "cycle_index": cyc, "issue": "short_cycle", "count": len(g)})
                            cp = p.with_name(p.name.replace("timeseries", "cycle_data"))
                            cap = pd.read_csv(cp)
                            q = pd.to_numeric(cap["Discharge_Capacity (Ah)"], errors="coerce")
                            cells.append({"dataset": name, "cell_id": cid, "charge_records": charge_count,
                                          "discharge_records": int((q > 0).sum()), "capacity_measurements": int(q.notna().sum()),
                                          "rated_capacity_ah": 1.35, "rated_capacity_basis": "CALCE CX2 family specification",
                                          "chemistry": "LiCoO2 cathode, trace Mn (CALCE CX2 family)",
                                          "ambient_temp_c": "25 (filename only, temperature channels empty)",
                                          "cycle_first": float(d.Cycle_Index.min()), "cycle_last": float(d.Cycle_Index.max()),
                                          "capacity_min_ah": float(q.min()), "capacity_max_ah": float(q.max()),
                                          "cell_temperature_available": bool(d["Cell_Temperature (C)"].notna().any()),
                                          "environment_temperature_available": bool(d["Environment_Temperature (C)"].notna().any()),
                                          "raw_charge_current_a": finite_range(d.loc[d["Current (A)"] > .03, "Current (A)"])})
                            for _, row in cap.loc[q.isna() | (q <= 0)].iterrows():
                                anomalies.append({"dataset": name, "cell_id": cid, "cycle_index": row.Cycle_Index, "issue": "missing_or_nonpositive_discharge_capacity"})
                            for _, row in cap.loc[(q > 0) & (q < .135)].iterrows():
                                anomalies.append({"dataset": name, "cell_id": cid, "cycle_index": row.Cycle_Index, "issue": "capacity_below_10pct_nominal_review"})
                elif p.suffix == ".mat":
                    entry["mat_index"] = whosmat(p)
                    data = loadmat(p, simplify_cells=True)
                    entry["keys"] = [k for k in data if not k.startswith("__")]
                    for cid, checkpoints in data.items():
                        if cid.startswith("__"):
                            continue
                        cap_values, time_checks, temp_values, currents = [], [], [], []
                        charge_n = discharge_n = 0
                        for ck, record in checkpoints.items():
                            cyc = int(ck.removeprefix("cyc"))
                            for mode, signal in record.items():
                                lengths = {k: len(np.atleast_1d(v)) for k, v in signal.items()}
                                t = np.asarray(signal["t"], float)
                                v = np.asarray(signal["v"], float)
                                q = np.asarray(signal["q"], float)
                                temp = np.asarray(signal["T"], float)
                                dt = np.diff(t)
                                cr = {"cell_id": cid, "cycle_index": cyc, "mode": mode, "keys": list(signal), "lengths": lengths,
                                      "time_value_range": finite_range(t), "voltage_range_v": finite_range(v),
                                      "charge_range_mah": finite_range(q), "temperature_range_c": finite_range(temp),
                                      "time_nonpositive_steps": int((dt <= 0).sum()),
                                      "nonfinite_count": int(sum((~np.isfinite(np.asarray(a, float))).sum() for a in signal.values()))}
                                curves.append(cr)
                                if cr["nonfinite_count"] or len(set(lengths.values())) != 1 or (dt <= 0).any():
                                    anomalies.append({"dataset": name, "cell_id": cid, "cycle_index": cyc, "mode": mode, "issue": "curve_nonfinite_length_or_time_issue"})
                                if mode == "C1ch":
                                    charge_n += 1
                                    if len(t) > 1 and (dt > 0).all():
                                        current = np.diff(q) / 1000 * 3600 / (dt * 86400)
                                        currents.extend([float(np.median(current)), float(np.min(current)), float(np.max(current))])
                                    temp_values.extend(finite_range(temp))
                                    time_checks.append(cyc)
                                if mode == "C1dc":
                                    discharge_n += 1
                                    cap_values.append(float(abs(q[-1] - q[0]) / 1000))
                        cells.append({"dataset": name, "cell_id": cid, "charge_records": charge_n,
                                      "discharge_records": discharge_n, "capacity_measurements": len(cap_values),
                                      "rated_capacity_ah": .740, "rated_capacity_basis": "Oxford authors' dataset paper, 740 mAh",
                                      "chemistry": "graphite / lithium cobalt oxide-lithium nickel cobalt oxide",
                                      "ambient_temp_c": "40 (authors' experimental documentation)",
                                      "cycle_first": min(time_checks), "cycle_last": max(time_checks),
                                      "capacity_min_ah": min(cap_values), "capacity_max_ah": max(cap_values),
                                      "cell_temperature_available": True, "environment_temperature_available": False,
                                      "derived_charge_current_a": finite_range(currents)})
                else:
                    entry["status"] = "nondata_file"
            except Exception as exc:
                entry["status"] = "unreadable"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                anomalies.append({"dataset": name, "source": entry["path"], "issue": entry["error"]})
            manifest.append(entry)
        print(f"Scanned {name}", flush=True)
    md = pd.read_csv(roots["nasa"] / "extracted/cleaned_dataset/metadata.csv")
    for cid, g in md.groupby("battery_id"):
        q = pd.to_numeric(g.loc[g.type.eq("discharge"), "Capacity"], errors="coerce")
        cells.append({"dataset": "nasa", "cell_id": cid, "charge_records": int(g.type.eq("charge").sum()),
                      "discharge_records": int(g.type.eq("discharge").sum()), "capacity_measurements": int(q.notna().sum()),
                      "rated_capacity_ah": 2., "rated_capacity_basis": "NASA documentation, rated 2 Ah",
                      "chemistry": "Li-ion 18650; cathode not specified in supplied readmes",
                      "ambient_temp_c": ",".join(map(str, sorted(g.loc[g.type.ne("impedance"), "ambient_temperature"].unique()))),
                      "cycle_first": 1, "cycle_last": int(g.type.eq("discharge").sum()),
                      "capacity_min_ah": float(q.min()), "capacity_max_ah": float(q.max()),
                      "cell_temperature_available": True, "environment_temperature_available": True})
    for _, r in md.loc[md.type.eq("discharge")].iterrows():
        q = pd.to_numeric(pd.Series([r.Capacity]), errors="coerce").iloc[0]
        if not np.isfinite(q) or q <= 0:
            anomalies.append({"dataset": "nasa", "cell_id": r.battery_id, "source": r.filename, "test_id": int(r.test_id), "issue": "missing_or_nonpositive_capacity"})
        elif q < .2:
            anomalies.append({"dataset": "nasa", "cell_id": r.battery_id, "source": r.filename, "test_id": int(r.test_id), "issue": "capacity_below_10pct_nominal_review"})
    on_disk = {p.name for p in (roots["nasa"] / "extracted/cleaned_dataset/data").glob("*.csv")}
    expected = set(md.filename)
    for fn in sorted(expected - on_disk):
        anomalies.append({"dataset": "nasa", "source": fn, "issue": "metadata_file_missing"})
    for fn in sorted(on_disk - expected):
        anomalies.append({"dataset": "nasa", "source": fn, "issue": "unreferenced_data_file"})
    for name, root in roots.items():
        save_json(OUT / f"{name}_source_metadata.json", json.loads((root / "dataset-metadata.json").read_text()))
    save_json(OUT / "file_manifest.json", manifest)
    save_json(OUT / "schemas.json", list(schema_groups.values()))
    save_json(OUT / "oxford_curve_manifest.json", curves)
    save_json(OUT / "anomalies.json", anomalies)
    pd.DataFrame(cells).sort_values(["dataset", "cell_id"]).to_csv(OUT / "cells.csv", index=False)
    save_json(OUT / "summary.json", {"files_scanned": len(manifest), "unreadable_files": sum(x["status"] == "unreadable" for x in manifest),
                                     "cells": dict(Counter(x["dataset"] for x in cells)), "anomaly_counts": dict(Counter(x["dataset"] + ":" + x["issue"] for x in anomalies))})
    print(pd.DataFrame(cells).groupby("dataset").agg(cells=("cell_id", "count"), charge_records=("charge_records", "sum"), capacity_measurements=("capacity_measurements", "sum"), min_capacity_ah=("capacity_min_ah", "min"), max_capacity_ah=("capacity_max_ah", "max")).to_string())
    print(json.dumps(Counter(x["dataset"] + ":" + x["issue"] for x in anomalies), indent=2))


if __name__ == "__main__":
    main()
