# Results

| file | contents |
|---|---|
| `operator_table.csv` | Table 1: recoverability and D with slide-level bootstrap intervals |
| `paired_deltas.csv` | paired slide-bootstrap ΔD within matched operator families |
| `pairlevel.csv` | per-predictor coupling of operator score to constituent-gene accuracy, 249,144 pair×slide×model observations |
| `threshold_sweep.csv` | coupling as a function of the hotspot quantile |
| `sensitivity.csv` | conditioner, baseline width, leave-one-model-out, CV partition, split-half reliability, recoverability cutoff |
| `audit_long.csv` | one row per slide × predictor × operator: score, permutation p, z, null SD, matched per-gene accuracy, tissue. 5,364 rows |
| `response_period/` | the tables behind the author responses, documented in its own README |

`audit_long.csv` is the input to `scripts/analyze.py` and to
`scripts/rebuttal_analyses.py`; every other table here is one of their outputs.
It is the only file a reader needs in order to regenerate the published tables
without re-running a predictor or a permutation null.
`scripts/make_synthetic_long.py` writes a table with the same schema and a
known ground truth, which exercises the pipeline but does not reproduce these
values.

Row counts differ by operator because three lineage operators run on 54 slides
rather than 61, and `progeny` carries 36 cells whose permutation test could not
be run under its pathway coverage gate. Those 36 are counted as non-recovery
rather than dropped; `distinctness.recoverability` documents why, and the
`n_untestable` column of `operator_table.csv` reports them.

Blank interval columns in `paired_deltas.csv` mark a contrast whose magnitude
member fails the recoverability gate. The point estimate is retained for
completeness; differencing against an operator with uninterpretable D says
nothing, so no interval is computed.
