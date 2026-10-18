"""Preregistered probability/alert evaluation; future truth is evaluation only."""
import json
import numpy as np
import pandas as pd
from .core import ROOT, OUT, BASE, write_json
from .tracking import risk_columns,rate_table

HORIZONS=[28,120,180]
P_GRID=[.50,.65,.80,.90,.95,.99,1.01]


def events_for(frame,eols,horizon,threshold,probability=True):
    rows=[]
    for _,e in eols[eols.censoring.eq('observed_crossing')].iterrows():
        h=frame[frame.dataset.eq(e.dataset)&frame.cell_id.eq(e.cell_id)].sort_values('cycle_index')
        if probability:
            alarms=h[h[f'probability_{horizon}']>=threshold]
        else:
            alarms=h[(h.predicted_rul_cycles<=horizon)&h.predicted_rul_cycles.notna()]
        first=float(alarms.cycle_index.iloc[0]) if len(alarms) else np.nan
        lead=e.eol_cycle-first
        category='no_eligible_history' if not len(h) else 'none'
        if len(alarms): category='premature' if lead>horizon else ('timely' if lead>0 else 'late')
        rows.append(dict(dataset=e.dataset,cell_id=e.cell_id,horizon_cycles=horizon,
            outcome=category,first_alert_cycle=first,lead_cycles=lead,lead_days=lead/2,
            has_timely_observation=bool(((e.eol_cycle-h.cycle_index>0)&(e.eol_cycle-h.cycle_index<=horizon)).any())))
    return pd.DataFrame(rows)


def choose_threshold(states,eols,horizon):
    """States/labels must come from the outer fold's inner validation cells."""
    if not eols.censoring.eq('observed_crossing').any():
        return 1.01,{'1.01':np.nan},'no_observed_inner_endpoint_abstain'
    scores={}
    for p in P_GRID:
        events=events_for(states,eols,horizon,p)
        costs=events.outcome.map({'timely':0,'premature':2,'late':3,'none':3,'no_eligible_history':3}).tolist()
        for _,e in eols[eols.censoring.eq('right_censored')].iterrows():
            h=states[states.dataset.eq(e.dataset)&states.cell_id.eq(e.cell_id)].sort_values('cycle_index')
            alerts=h[h[f'probability_{horizon}']>=p]
            if len(alerts): costs.append(2. if alerts.cycle_index.iloc[0]+horizon<=e.last_reference_cycle else 0.)
            else: costs.append(0.)
        scores[str(p)]=float(np.mean(costs)) if costs else np.inf
    minimum=min(scores.values())
    winner=max(p for p in P_GRID if scores[str(p)]==minimum)
    return winner,scores,'minimum_predeclared_event_cost'


def event_summary(events):
    rows=[]
    for key,g in events.groupby(['method','validation','dataset','horizon_cycles']):
        record=dict(zip(['method','validation','dataset','horizon_cycles'],key),events=len(g))
        for outcome in ['timely','premature','late','none','no_eligible_history']: record[outcome]=int(g.outcome.eq(outcome).sum())
        lead=g[g.first_alert_cycle.notna()].lead_days
        record.update(timely_rate=record['timely']/len(g),events_with_timely_observation=int(g.has_timely_observation.sum()),
            lead_days_q25=float(lead.quantile(.25)),lead_days_median=float(lead.median()),lead_days_q75=float(lead.quantile(.75)))
        rows.append(record)
    return pd.DataFrame(rows)


def run():
    out=OUT/'E6_probabilistic'; out.mkdir(exist_ok=True)
    outer=risk_columns(pd.read_parquet(OUT/'E5_filter/predictions.parquet'))
    inner=pd.read_parquet(OUT/'E5_filter/inner_validation_states.parquet')
    eols=pd.read_csv(BASE/'end_of_life.csv'); eols['group_id']=eols.dataset+':'+eols.cell_id
    thresholds=[]; all_events=[]; parts=[]
    for (protocol,fold),group in outer.groupby(['validation','fold']):
        vi=risk_columns(inner[inner.validation.eq(protocol)&inner.fold.eq(fold)])
        truth=eols[eols.group_id.isin(vi.group_id)]
        o=group.copy()
        for h in HORIZONS:
            p,scores,reason=choose_threshold(vi,truth,h)
            o[f'alert_threshold_{h}']=p; o[f'probability_alert_{h}']=o[f'probability_{h}']>=p
            thresholds.append(dict(validation=protocol,fold=fold,horizon=h,p_star=p,inner_costs=scores,reason=reason,
                selection_cells=sorted(vi.group_id.unique()),outer_test_cells=sorted(group.group_id.unique())))
        parts.append(o)
    outer=pd.concat(parts,ignore_index=True)
    # Explicit event rows for all 19 crossings, including cells with no eligible history.
    for protocol in ['leave_one_cell_out','leave_one_dataset_out']:
        p=outer[outer.validation.eq(protocol)]
        for h in HORIZONS:
            event_parts=[]
            for _,truth in eols[eols.censoring.eq('observed_crossing')].iterrows():
                cell=p[p.dataset.eq(truth.dataset)&p.cell_id.eq(truth.cell_id)]
                threshold=float(cell[f'alert_threshold_{h}'].iloc[0]) if len(cell) else 1.01
                event_parts.append(events_for(cell,truth.to_frame().T,h,threshold))
            events=pd.concat(event_parts,ignore_index=True); events['validation']=protocol; events['method']='probabilistic'
            all_events.append(events)
    old=pd.read_csv(ROOT/'reports/review/alarm_observations.csv'); old=old[old.signal.eq('raw')].rename(columns={'today_cycle':'cycle_index'})
    for protocol in ['leave_one_cell_out','leave_one_dataset_out']:
        for h in HORIZONS:
            events=events_for(old[old.validation.eq(protocol)],eols,h,0,False)
            events['validation']=protocol; events['method']='frozen_raw_linear'; all_events.append(events)
    events=pd.concat(all_events,ignore_index=True)
    events.to_csv(out/'alert_events.csv',index=False); event_summary(events).to_csv(out/'alert_metrics.csv',index=False)
    write_json(out/'threshold_audit.json',thresholds)
    outer.to_parquet(out/'predictions.parquet',index=False)
    # Posterior RUL intervals at fixed retrospective life-stage anchors.
    anchors=[]
    for protocol in ['leave_one_cell_out','leave_one_dataset_out']:
        for _,truth in eols[eols.censoring.eq('observed_crossing')].iterrows():
            p=outer[outer.validation.eq(protocol)&outer.dataset.eq(truth.dataset)&outer.cell_id.eq(truth.cell_id)]
            for stage,fraction in [('early',.2),('mid',.5),('late',.85)]:
                today=truth.first_reference_cycle+fraction*(truth.eol_cycle-truth.first_reference_cycle)
                history=p[p.cycle_index<=today].sort_values('cycle_index')
                if not len(history):
                    for level in [50,80,95]: anchors.append(dict(validation=protocol,dataset=truth.dataset,cell_id=truth.cell_id,stage=stage,nominal=level/100,available=False))
                    continue
                last=history.iloc[-1]; actual=truth.eol_cycle-last.cycle_index
                for level in [50,80,95]:
                    low=last[f'rul_lower_{level}']; high=last[f'rul_upper_{level}']
                    anchors.append(dict(validation=protocol,dataset=truth.dataset,cell_id=truth.cell_id,stage=stage,nominal=level/100,
                        available=True,lower_cycles=low,upper_cycles=high,actual_cycles=actual,
                        covered=bool(low<=actual<=high),finite_upper=bool(np.isfinite(high)),
                        median_absolute_error_cycles=abs(last.rul_median-actual)))
    a=pd.DataFrame(anchors); a.to_csv(out/'rul_interval_anchors.csv',index=False)
    cal=[]
    for key,g in a.groupby(['validation','dataset','stage','nominal']):
        available=g[g.available]; finite=available[available.finite_upper.eq(True)] if len(available) else available
        cal.append(dict(zip(['validation','dataset','stage','nominal'],key),requested=len(g),available=len(available),
            coverage=float(available.covered.mean()) if len(available) else np.nan,
            finite_upper_fraction=float(available.finite_upper.mean()) if len(available) else np.nan,
            finite_mean_width_cycles=float((finite.upper_cycles-finite.lower_cycles).mean()) if len(finite) else np.nan))
    pd.DataFrame(cal).to_csv(out/'rul_interval_calibration.csv',index=False)
    brier=[]; reliability=[]
    for (protocol,dataset),g in outer.groupby(['validation','dataset']):
        truth=eols[eols.dataset.eq(dataset)].set_index('cell_id')
        for h in HORIZONS:
            cases=[]
            for _,row in g.iterrows():
                e=truth.loc[row.cell_id]; actual=np.nan
                if e.censoring=='observed_crossing' and row.cycle_index<e.eol_cycle: actual=float(e.eol_cycle-row.cycle_index<=h)
                elif e.censoring=='right_censored' and row.cycle_index+h<=e.last_reference_cycle: actual=0.
                if np.isfinite(actual): cases.append((row.cell_id,row[f'probability_{h}'],actual))
            c=pd.DataFrame(cases,columns=['cell_id','probability','actual'])
            if not len(c): continue
            c['squared_error']=(c.probability-c.actual)**2
            brier.append(dict(validation=protocol,dataset=dataset,horizon=h,n_rows=len(c),n_cells=c.cell_id.nunique(),
                pooled_brier=float(c.squared_error.mean()),cell_macro_brier=float(c.groupby('cell_id').squared_error.mean().mean())))
            c['bin']=pd.cut(c.probability,[-.001,.2,.4,.6,.8,1.001])
            for label,part in c.groupby('bin',observed=True): reliability.append(dict(validation=protocol,dataset=dataset,horizon=h,bin=str(label),n=len(part),
                    mean_probability=float(part.probability.mean()),observed_frequency=float(part.actual.mean())))
    pd.DataFrame(brier).to_csv(out/'brier.csv',index=False); pd.DataFrame(reliability).to_csv(out/'reliability.csv',index=False)
    rates=rate_table(pd.read_parquet(ROOT/'data/processed/capacity_references.parquet'),set(eols.group_id))
    rates.to_csv(out/'reference_rates.csv',index=False)
    base=pd.read_csv(BASE/'metrics_per_dataset.csv'); base=base[base.model.eq('training_selected')&base.variant.eq('no_temperature')]
    scales=[]
    for dataset,g in rates.groupby('dataset'):
        rate=float(g[g.rate_per_cycle>0].rate_per_cycle.median())
        for protocol in ['leave_one_cell_out','leave_one_dataset_out']:
            b=base[base.validation.eq(protocol)&base.dataset.eq(dataset)].macro_cell_mae_raw_pp.iloc[0]/100
            p=outer[outer.validation.eq(protocol)&outer.dataset.eq(dataset)]
            err=float(p.assign(error=(p.prediction-p.soh).abs()).groupby('cell_id').error.mean().mean())
            scales.append(dict(dataset=dataset,validation=protocol,median_positive_rate_soh_per_cycle=rate,
                baseline_implied_error_cycles=b/rate,filter_implied_error_cycles=err/rate,
                filter_implied_error_scenario_days=err/rate/2))
    pd.DataFrame(scales).to_csv(out/'rul_error_scale.csv',index=False)
    print(event_summary(events).query("validation == 'leave_one_cell_out'").to_string(index=False),flush=True)
