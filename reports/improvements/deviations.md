# Deviations and implementation log

No experiments have run at preregistration. Scientific changes must retain both
original and revised results. Implementation corrections without changed design
will be dated and explained here before rerunning the affected stage.

## 2026-10-18 — reporting implementation corrections

The first report-generation attempt stopped before producing results.md because
a loop variable shadowed the dataset-metrics table. Rename the loop variable and
rerun report assembly; no prediction, threshold, feature, split or fitted model
changes. E9 aggregate supplementary median metrics now call the existing causal
median instead of aliasing raw predictions; per-target E9 metrics already used
the correct median. No original/revised scientific configurations exist: all
preregistered configurations and raw results remain unchanged.

The fleet-count experiment was separately preregistered and inserted after E1
at the user's request. It uses its frozen point-RUL fallback rather than later
choosing E6 probabilities based on the fleet-count test results.

Entry-relative feature outputs now explicitly mark the first K warmup rows as
unavailable (NaN), enforcing their prediction-time contract. These rows were
already excluded from every delta/combined/ICA training and scoring set under
the preregistration; all post-warmup feature values and reported predictions
are unchanged. A test checks both unavailability and the unchanged later deltas.
