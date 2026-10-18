"""Causal Gaussian state tracking with explicitly training-only noise fitting."""
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.model_selection import GroupKFold
from battery_analytics.rul import reference_histories
from .core import fit, macro_mae

Q_GRID=[.0001,.001,.01,.05]
ZS=norm.ppf(np.clip(qmc.Sobol(2,scramble=True,seed=42).random_base2(9),1e-9,1-1e-9))


def rate_table(references, cells):
    r=reference_histories(references)
    rows=[]
    for (dataset,cell),h in r.groupby(['dataset','cell_id']):
        gid=dataset+':'+cell
        if gid not in cells or len(h)<2 or np.ptp(h.cycle_index)<=0: continue
        slope=float(np.polyfit(h.cycle_index,h.soh,1)[0])
        rows.append(dict(group_id=gid,dataset=dataset,cell_id=cell,rate_per_cycle=-slope,
                         span=float(np.ptp(h.cycle_index)),n_references=len(h)))
    return pd.DataFrame(rows)


def train_prior(references, cells):
    rates=rate_table(references,cells)
    scale=max(1.,float(rates.span.median()))
    slopes=-rates.rate_per_cycle.to_numpy()*scale
    return dict(time_scale=scale, rate_mean=float(np.median(slopes)),
                rate_variance=max(1e-6,float(np.var(slopes))), prior_cells=sorted(rates.group_id.tolist()))


def filter_cell(cycles, observations, parameters):
    """Input observations are model estimates only. True capacity is inaccessible."""
    x=np.asarray(cycles,float); z=np.asarray(observations,float)
    R=parameters['observation_variance']; scale=parameters['time_scale']; q=parameters['process_std']
    m=np.array([z[0],parameters['rate_mean']]); P=np.diag([R,parameters['rate_variance']])
    result=[]
    for j in range(len(x)):
        if j:
            dt=max(1e-6,(x[j]-x[j-1])/scale)
            F=np.array([[1.,dt],[0.,1.]])
            Q=q*q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
            m=F@m; P=F@P@F.T+Q
            gain=P[:,0]/(P[0,0]+R)
            m=m+gain*(z[j]-m[0])
            # Joseph form preserves a positive covariance numerically.
            A=np.eye(2)-np.outer(gain,[1.,0.]); P=A@P@A.T+R*np.outer(gain,gain)
            P=(P+P.T)/2
        result.append([m[0],m[1]/scale,P[0,0],P[0,1]/scale,P[1,1]/scale**2])
    return np.asarray(result)


def track(frame, observation_predictions, parameters):
    a=frame.copy(); a['observation_prediction']=np.asarray(observation_predictions)
    a=a.sort_values(['group_id','cycle_index','charge_end_time_s'])
    outputs=[]
    for _,g in a.groupby('group_id',sort=False):
        state=filter_cell(g.cycle_index,g.observation_prediction,parameters)
        h=g.copy()
        for j,c in enumerate(['prediction','rate_per_cycle','variance_soh','covariance_soh_rate','variance_rate']): h[c]=state[:,j]
        for level in [50,80,95]:
            width=norm.ppf((1+level/100)/2)*np.sqrt(h.variance_soh)
            h[f'lower_{level}']=h.prediction-width; h[f'upper_{level}']=h.prediction+width
        outputs.append(h)
    return pd.concat(outputs).sort_index()


def fit_noise(inner_training, columns, references):
    """Only inner-training cells enter these cross-fits, labels, priors and q search."""
    oof=pd.Series(index=inner_training.index,dtype=float)
    splits=[]
    cv=GroupKFold(n_splits=3)
    for ti,vi in cv.split(inner_training,groups=inner_training.group_id):
        tr,va=inner_training.iloc[ti],inner_training.iloc[vi]
        model=fit(tr,columns); oof.loc[va.index]=model.predict(va[columns])
        splits.append(dict(train_cells=sorted(tr.group_id.unique()),validation_cells=sorted(va.group_id.unique())))
    residual=(oof-inner_training.soh)**2
    R=max(1e-6,float(pd.DataFrame({'cell':inner_training.group_id,'se':residual}).groupby('cell').se.mean().mean()))
    parameters=train_prior(references,set(inner_training.group_id))
    parameters.update(observation_variance=R)
    scores={}
    for q in Q_GRID:
        p=track(inner_training,oof,dict(parameters,process_std=q))
        scores[str(q)]=macro_mae(p,p.prediction)
    minimum=min(scores.values())
    q=next(q for q in Q_GRID if scores[str(q)]<=minimum+.20)
    return dict(parameters,process_std=q,noise_training_cells=sorted(inner_training.group_id.unique()),
                noise_crossfit_splits=splits,process_scores_pp=scores)


def crossing_samples(states):
    """Fixed posterior draws; no future capacity or endpoint input."""
    mu=states.prediction.to_numpy()[:,None]; rate=states.rate_per_cycle.to_numpy()[:,None]
    var=np.maximum(states.variance_soh.to_numpy(),1e-15)[:,None]
    cross=states.covariance_soh_rate.to_numpy()[:,None]
    var_rate=np.maximum(states.variance_rate.to_numpy(),0)[:,None]
    soh=mu+np.sqrt(var)*ZS[None,:,0]
    beta=rate+cross/np.sqrt(var)*ZS[None,:,0]+np.sqrt(np.maximum(var_rate-cross**2/var,0))*ZS[None,:,1]
    with np.errstate(divide='ignore',invalid='ignore'):
        remaining=np.where(beta<0,(.7-soh)/beta,np.inf)
    return np.where(soh<=.7,0.,np.maximum(remaining,0.))


def risk_columns(states,horizons=(28,120,180)):
    result=states.copy(); draws=crossing_samples(result)
    for horizon in horizons: result[f'probability_{horizon}']=(draws<=horizon).mean(axis=1)
    # Inverse empirical CDF retains infinite bounds (np.quantile would inf-inf).
    ordered=np.sort(draws,axis=1)
    for level in [50,80,95]:
        tail=(1-level/100)/2
        for label,p in [('lower',tail),('upper',1-tail)]:
            result[f'rul_{label}_{level}']=ordered[:,int(np.ceil(p*512))-1]
    result['rul_median']=ordered[:,255]
    return result
