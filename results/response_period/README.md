# Response-period results

Everything in this directory was computed during the ML4H 2026 author-response
period, after the submitted version. Each file is the table behind a specific
reviewer request. The files in the parent directory are the submitted results.

| file | reviewer request | produced by |
|---|---|---|
| `reordering.csv` | does the operator change which model is selected | `analyses.discrimination_table` |
| `per_tissue_delta.csv` | is the contrast consistent across tissues | `analyses.per_tissue_delta` |
| `out_of_tissue.csv` | hold out the two dominant cancers entirely | `analyses.out_of_tissue` |
| `leave_one_tissue_out_range.csv` | does any single tissue carry the result | `analyses.leave_one_tissue_out` |
| `regime_split.csv` | do the six predictors span different information regimes | `analyses.regime_split` |
| `conditioner_sweep.csv` | does the contrast depend on the conditioning model | `analyze.sensitivity`, extended to MCP |
| `reliability.csv` | is the residual reproducible, and the same test for MCP | `reliability.residualized_split_half`, `repeated_split_half` |
| `block_null.csv` | a spatially constrained randomization | `spatial_nulls.block_perturb` at 25 blocks |
| `error_conditioned_D.csv` | condition on error magnitude and error smoothness | `errors.error_conditioned_D` |
| `error_features_summary.csv` | descriptives of the error features above | `errors.error_features` |

## Reading the two conditioner provenances

`conditioner_sweep.csv` carries a `provenance` column. The LR rows marked
`submitted` are the values behind the one-line range reported in the
submission. The MCP rows marked `response period` are new, since the
submission swept the conditioner for the LR family only.

A rerun during the response period produced different non-spline LR values
(linear 0.360, rf 0.384, gbm 0.374, knn 0.344) while reproducing the spline
value exactly in both families. The two runs differ in how the non-spline
conditioners are specified, not in the data. The submitted values are the ones
reported here, because those are the ones the paper states.

## Reading `error_conditioned_D.csv`

The `accuracy only (reported)` rows use a spline conditioner on one feature.
The added-feature rows use a linear conditioner on standardised inputs,
because a spline basis on three features multiplies out. The rows are
therefore comparable with each other and not directly with the spline row,
which is why `conditioner` is a column. What is comparable throughout is the
direction each operator's D moves: `D_mag` falls and `D_loc` holds or rises in
every refit.

## Reading `reliability.csv`

Two different tests share the file. The LR rows report the reliability of the
accuracy-conditioned residual, which is the quantity D is computed from, with
the raw aggregate reliability alongside for contrast. The MCP rows report a
marker-panel split-half reliability and the resulting correction for
attenuation. Blank cells mean that test was not run for that operator, not
that it returned nothing.
