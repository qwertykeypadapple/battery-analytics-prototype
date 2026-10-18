"""Small fixed survival models, causal history features and censoring weights."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SURVIVAL_METHODS = ('km', 'hazard', 'hazard_known_history', 'cox')
PROB_HORIZONS = (60, 120, 180)
ALERT_HORIZONS = (28, 120, 180)
ALL_HORIZONS = (28, 60, 120, 180)
THRESHOLDS = (.50, .65, .80, .90, .95, .99, 1.01)
PERIOD = 10.


class History:
    """No true labels: last raw SoH and OLS fade over last 12 distinct points."""
    def __init__(self, frame):
        h = frame[['cycle_index', 'charge_end_time_s', 'prediction']].copy()
        h = h.sort_values(['cycle_index', 'charge_end_time_s']).drop_duplicates('cycle_index', keep='last')
        self.x = h.cycle_index.to_numpy(float)
        self.soh = h.prediction.to_numpy(float)
        self.rate = np.full(len(h), np.nan)
        for i in range(5, len(h)):
            a = max(0, i - 11)
            x, y = self.x[a:i+1], self.soh[a:i+1]
            if np.isfinite(y).all() and np.ptp(x) > 0:
                relative = x - x[-1]
                self.rate[i] = -np.dot(relative-relative.mean(), y-y.mean()) / np.dot(relative-relative.mean(), relative-relative.mean())

    def at(self, ages):
        ages = np.atleast_1d(ages).astype(float)
        features = np.full((len(ages), 3), np.nan)
        last = np.full(len(ages), np.nan)
        if len(self.x):
            idx = np.searchsorted(self.x, ages, side='right') - 1
            valid = idx >= 0
            features[valid, 0] = self.soh[idx[valid]]
            features[valid, 1] = self.rate[idx[valid]]
            features[valid, 2] = ages[valid]
            last[valid] = self.x[idx[valid]]
        return features, last


class KaplanMeier:
    """One event/censor per subject with delayed entry; step survival function."""
    def __init__(self, endpoints, censoring=False):
        good = endpoints[endpoints.censoring.isin(['observed_crossing', 'right_censored'])]
        self.entry = good.first_reference_cycle.to_numpy(float)
        self.stop = np.where(good.censoring.eq('observed_crossing'), good.eol_cycle, good.last_reference_cycle).astype(float)
        event = good.censoring.eq('observed_crossing').to_numpy()
        if censoring:
            event = ~event
        self.max_time = float(self.stop.max()) if len(self.stop) else -np.inf
        self.min_entry = float(self.entry.min()) if len(self.entry) else np.inf
        times, surv, value = [], [], 1.
        for t in np.unique(self.stop[event]):
            n = np.sum((self.entry <= t) & (self.stop >= t))
            d = np.sum((self.stop == t) & event)
            value *= 1 - d / n
            times.append(t); surv.append(value)
        self.times, self.survival = np.asarray(times), np.asarray(surv)
        self.status = 'fitted' if len(good) else 'no_training_exposure'

    def at(self, times, left=False):
        times = np.atleast_1d(times).astype(float)
        idx = np.searchsorted(self.times, times, side='left' if left else 'right') - 1
        out = np.ones(len(times))
        valid = idx >= 0
        out[valid] = self.survival[idx[valid]]
        return out

    def risk(self, ages, horizon, features=None):
        ages = np.atleast_1d(ages).astype(float)
        s0, sh = self.at(ages), self.at(ages + horizon)
        valid = (ages >= self.min_entry) & (ages + horizon <= self.max_time) & (s0 > 1e-12)
        out = np.full(len(ages), np.nan)
        out[valid] = np.clip(1 - sh[valid] / s0[valid], 0, 1)
        return out


def person_periods(histories, endpoints):
    rows = []
    for e in endpoints.itertuples():
        if e.censoring not in ('observed_crossing', 'right_censored') or e.group_id not in histories:
            continue
        h = histories[e.group_id]
        valid = np.flatnonzero(np.isfinite(h.rate) & np.isfinite(h.soh) & (h.x >= e.first_reference_cycle))
        stop = e.eol_cycle if e.censoring == 'observed_crossing' else e.last_reference_cycle
        if not len(valid) or h.x[valid[0]] >= stop:
            continue
        starts = np.arange(h.x[valid[0]], stop, PERIOD)
        features, last = h.at(starts)
        for t, x, seen in zip(starts, features, last):
            if not np.isfinite(x).all():
                continue
            end = min(t + PERIOD, stop)
            event = e.censoring == 'observed_crossing' and end == stop
            rows.append(dict(group_id=e.group_id, start=t, stop=end, event=int(event),
                             charge_soh=x[0], fade_rate=x[1], cycle=t, last_charge_cycle=seen,
                             logistic_eligible=bool(event or end - t >= PERIOD - 1e-8)))
    return pd.DataFrame(rows, columns=['group_id', 'start', 'stop', 'event', 'charge_soh', 'fade_rate',
                                       'cycle', 'last_charge_cycle', 'logistic_eligible'])


class Hazard:
    def __init__(self, periods, known_history=False):
        self.columns = ['charge_soh', 'fade_rate'] + (['cycle'] if known_history else [])
        self.dim = len(self.columns)
        p = periods[periods.logistic_eligible]
        self.n_events, self.n_rows = int(p.event.sum()), len(p)
        self.status, self.detail = 'no_training_exposure', ''
        if not len(p):
            return
        self.scaler = StandardScaler().fit(p[self.columns].to_numpy(float))
        if not self.n_events:
            self.status = 'no_training_events'
            return
        if p.event.nunique() < 2:
            self.status = 'failed'; self.detail = 'no_non_event_period'
            return
        try:
            self.model = LogisticRegression(C=1., penalty='l2', solver='lbfgs', max_iter=2000, tol=1e-9)
            self.model.fit(self.scaler.transform(p[self.columns].to_numpy(float)), p.event)
            self.status = 'fitted' if self.model.n_iter_.max() < 2000 else 'failed'
            self.detail = f'iterations={self.model.n_iter_.max()}'
        except (ValueError, FloatingPointError) as exc:
            self.status, self.detail = 'failed', str(exc)

    def risk(self, ages, horizon, features):
        features = np.asarray(features, float)[:, :self.dim]
        out = np.full(len(features), np.nan)
        valid = np.isfinite(features).all(axis=1)
        if self.status == 'no_training_events':
            out[valid] = 0.
        elif self.status == 'fitted' and valid.any():
            p = self.model.predict_proba(self.scaler.transform(features[valid]))[:, 1]
            with np.errstate(divide='ignore'):
                out[valid] = -np.expm1(np.log1p(-p) * horizon / PERIOD)
        return out


class Cox:
    def __init__(self, periods):
        self.status, self.detail = 'no_training_exposure', ''
        self.n_rows, self.n_events = len(periods), int(periods.event.sum())
        self.max_time = float(periods.stop.max()) if len(periods) else -np.inf
        self.event_times, self.baseline_jumps = np.array([]), np.array([])
        if not len(periods):
            return
        self.scaler = StandardScaler().fit(periods[['charge_soh', 'fade_rate']].to_numpy(float))
        if not self.n_events:
            self.status = 'no_training_events'; self.beta = np.zeros(2)
            return
        x = self.scaler.transform(periods[['charge_soh', 'fade_rate']].to_numpy(float))
        starts, stops, events = periods.start.to_numpy(), periods.stop.to_numpy(), periods.event.to_numpy(bool)
        terms = []
        for t in np.unique(stops[events]):
            risk = (starts < t) & (stops >= t)
            event = events & (stops == t)
            terms.append((float(t), x[risk], x[event].sum(axis=0), int(event.sum())))

        def objective(beta):
            value, grad = .5 * np.dot(beta, beta), beta.copy()
            for _, rx, event_sum, d in terms:
                eta = rx @ beta
                denom = logsumexp(eta)
                value += (d * denom - np.dot(event_sum, beta)) / self.n_events
                grad += (d * (np.exp(eta - denom) @ rx) - event_sum) / self.n_events
            return value, grad
        result = minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B',
                          options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8))
        self.status = 'fitted' if result.success else 'failed'
        self.detail = str(result.message)
        self.beta = result.x
        self.event_times = np.asarray([t for t, _, _, _ in terms])
        self.baseline_jumps = np.asarray([d / np.exp(logsumexp(rx @ self.beta)) for _, rx, _, d in terms])

    def cumulative(self, times):
        idx = np.searchsorted(self.event_times, times, side='right') - 1
        out = np.zeros(len(np.atleast_1d(times)))
        valid = idx >= 0
        out[valid] = np.cumsum(self.baseline_jumps)[idx[valid]]
        return out

    def risk(self, ages, horizon, features):
        ages, features = np.atleast_1d(ages).astype(float), np.asarray(features, float)[:, :2]
        valid = np.isfinite(features).all(axis=1) & (ages + horizon <= self.max_time)
        out = np.full(len(ages), np.nan)
        if self.status == 'no_training_events':
            out[valid] = 0.
        elif self.status == 'fitted' and valid.any():
            eta = self.scaler.transform(features[valid]) @ self.beta
            base = self.cumulative(ages[valid] + horizon) - self.cumulative(ages[valid])
            # Avoid 0*inf in numerically extreme transfer without capping risk.
            with np.errstate(over='ignore'):
                cumulative = np.where(base > 0, np.exp(np.log(np.maximum(base, 1e-300)) + eta), 0.)
            out[valid] = -np.expm1(-cumulative)
        return out


@dataclass
class Bundle:
    models: dict
    censoring: KaplanMeier
    periods: pd.DataFrame


def fit_bundle(histories, endpoints):
    periods = person_periods(histories, endpoints)
    return Bundle(dict(km=KaplanMeier(endpoints), hazard=Hazard(periods),
                       hazard_known_history=Hazard(periods, True), cox=Cox(periods)),
                  KaplanMeier(endpoints, censoring=True), periods)


def ipcw(censoring, ages, stop, event, horizon):
    """Training-only conditional censoring weights; unknown outcome weight is 0."""
    ages = np.atleast_1d(ages).astype(float)
    elapsed = stop - ages
    y = np.where(event, (elapsed <= horizon).astype(float), np.where(elapsed >= horizon, 0., np.nan))
    event_by_h = bool(event) & (elapsed <= horizon)
    evaluated_time = np.where(event_by_h, stop, ages + horizon)
    gt = censoring.at(ages)
    gu = np.where(event_by_h, censoring.at(evaluated_time, left=True), censoring.at(evaluated_time))
    conditional = np.divide(gu, gt, out=np.full(len(ages), np.nan), where=gt > 0)
    known = np.isfinite(y)
    support = (ages <= censoring.max_time) & (gt > 0) & (~known | ((evaluated_time <= censoring.max_time) & (conditional >= .05)))
    weight = np.where(known & support, 1 / np.maximum(conditional, 1e-300), 0.)
    return y, weight, support


def first_alert(probability_frame, horizon, threshold):
    h = probability_frame[probability_frame[f'p_{horizon}'].ge(threshold)]
    return float(h.cycle_index.min()) if len(h) else np.nan


def event_class(first, endpoint, horizon):
    if not np.isfinite(first):
        return 'none'
    lead = endpoint - first
    return 'premature' if lead > horizon else ('timely' if lead > 0 else 'late')
