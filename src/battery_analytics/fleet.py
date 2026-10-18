from __future__ import annotations
import numpy as np
import pandas as pd
from .rul import forecast


def fleet_snapshot(predictions, inventory, today, rul_config, *, value='prediction', method='linear',
                   watch_soh=.80, replace_days=14, watch_days=45):
    records = []
    for _, cell in inventory.iterrows():
        h = predictions[predictions.dataset.eq(cell.dataset) & predictions.cell_id.eq(cell.cell_id)]
        result = forecast(h, today, rul_config, method, value)
        soh, days = result['current_predicted_soh'], result['predicted_rul_days']
        reason = result['forecast_reason']
        if not np.isfinite(soh):
            status, evidence = 'No data', 'No eligible charge at or before selected cycle'
        elif today > h.cycle_index.max():
            status, evidence = 'History ended', 'Selected cycle is beyond this cell’s eligible recorded history'
        elif not 0 < soh <= 1.5:
            status, evidence = 'Check estimate', 'Model estimate is outside review bounds'
        elif soh <= rul_config['end_of_life_soh'] or (np.isfinite(days) and days <= replace_days):
            status, evidence = 'Replace soon', 'Predicted health or scenario RUL crosses alert threshold'
        elif soh <= watch_soh or not np.isfinite(days) or days <= watch_days:
            status, evidence = 'Watch', reason if not np.isfinite(days) else 'Predicted health or scenario RUL crosses watch threshold'
        else:
            status, evidence = 'Healthy', 'No configured threshold crossed in this laboratory estimate'
        records.append(dict(dataset=cell.dataset, cell=cell.cell_id, as_of_cycle=result['as_of_cycle'],
                            soh_pct=100 * soh, cycles_left=result['predicted_rul_cycles'], scenario_days_left=days,
                            status=status, evidence=evidence, forecast_reason=reason))
    return pd.DataFrame(records)
