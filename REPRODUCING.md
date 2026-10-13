# Reproducing

There are four levels here, and they cost very different amounts. Levels 0 and
1 need nothing but a clone. Level 2 needs one file. Level 3 needs a GPU, a
HuggingFace token and about fifteen hours.

Read this first, because it says plainly what this repository can and cannot
regenerate on its own.

## What is and is not in the clone

In:

- `audit/` — the method, as readable definitions
- `scripts/` — the two drivers, plus the response-period analyses and the
  synthetic generator
- `tests/` — 56 invariants, no data required
- `results/` — the derived tables behind every reported number
- `results/audit_long.csv` — the per-cell scores, 5,364 rows, that every other
  table is computed from
- `resources/slide_tissue.csv` — the slide-to-tissue map the tissue and regime
  analyses need
- `results/response_period/` — the tables behind the author responses
- `docs/datasets.md` — how to connect the pipeline to HEST-1k, including the
  cache contract and the four invariants nothing downstream checks for you
- `docs/predictors.md` — the six predictors, including the Phikon encoder that
  Ridge, PLS and kNN depend on
- `docs/anonymous-release.md` — releasing this for double-blind review

The Colab notebook that produced the published numbers is not released. It is
tied to one Drive layout and is not runnable by a reader.
`results/audit_long.csv` is what stands in for it: every table in the paper is
computed from that file, so a reported figure can be checked against the same
per-cell data it was built from.

Not in, and why:

| missing | why | needed for |
|---|---|---|
| `resources/*.csv`, `*.json` | LIANA, PROGENy and MCP-counter panels are third-party; redistribution is not ours to grant | level 3 |
| the prediction cache | roughly 14 GB of predicted expression over 61 slides and 6 predictors, derived from HEST-Bench under its own terms | level 3 |

The tests do not need any of it. They stub the panels, so a clone with nothing
added still verifies the implementation.

## Level 0 — the invariants. No data, under a minute.

```bash
pip install -r requirements.txt
pytest tests/ -q
```

56 tests. They pin the properties whose violation would silently change a
reported number rather than raise: the permutation p-value floor at each B, the
bootstrap relabelling, the sign of D on inputs whose answer is known
analytically, that the domain graph is destroyed by shuffling coordinates, that
each operator's null destroys the structure that operator measures, that the
block permutation preserves autocorrelation while a spot shuffle does not, and
that the correction for attenuation only ever lowers D.

This is the level at which a reader can check that the method is what the paper
says it is. It does not reproduce any number in the paper.

## Level 1 — the pipeline end to end on synthetic data. No data, about ten minutes.

```bash
python scripts/make_synthetic_long.py --out results/synthetic_long.csv
python scripts/analyze.py --long results/synthetic_long.csv --out /tmp/syn \
       --tissue-map results/synthetic_slide_tissue.csv
python scripts/rebuttal_analyses.py --long results/synthetic_long.csv \
       --tissue-map results/synthetic_slide_tissue.csv --out /tmp/syn
```

`make_synthetic_long.py` fabricates a long table with the real schema, the real
cohort shape (61 slides over six tissues, six predictors, fifteen operators)
and a known ground truth: each operator's score is a mixture of a monotone
function of accuracy and independent noise, so the true D is one minus the
mixing weight by construction. The weights are set from the published D values.

**The numbers it produces are not the paper's numbers and will not match
them.** What it reproduces is the structure. Specifically, the analysis half of
the pipeline sorts the fifteen operators into the same three groups with the
same membership as Table 1, and the redundant operators reorder fewer model
pairs than the distinct ones. Both of those are asserted in
`tests/test_response_period.py` so they cannot drift silently.

This is the level at which a reader can check that the code runs, that the
estimators recover a ground truth they can see, and that a change to the
library moves the numbers it should move.

## Level 2 — the published tables. One file, about twenty minutes.

With `results/audit_long.csv` in place:

```bash
python scripts/analyze.py --long results/audit_long.csv --out results/ \
       --tissue-map resources/slide_tissue.csv
python scripts/rebuttal_analyses.py --long results/audit_long.csv \
       --tissue-map resources/slide_tissue.csv --out results/response_period/
```

This regenerates every derived table in `results/` from the per-cell scores,
without touching a predictor or re-running a permutation null. It is the level
most readers want, and it is the only level at which the published values come
back.

`audit_long.csv` is one row per slide x predictor x operator, 5,490 rows, with
columns:

```
slide predictor operator family score p z null_sd B matched_pcc
n_matched_genes n_spots
```

plus `pcc_mean pcc_median pcc_sd pcc_q10 pcc_q90 pcc_iqr pcc_frac_pos
pcc_skew` if the rich baseline was computed, and `err_rmse err_mae err_moran`
if the error features were.

### Exporting it from the notebook

The audit frame in the notebook is already this table under another name. In
the notebook, after the audit run:

```python
keep = ["slide", "model", "operator", "family", "score", "p", "z",
        "null_sd", "B", "matched_pcc", "n_matched_genes", "n_spots"]
rich = [c for c in audit_c.columns if c.startswith(("pcc_", "err_"))]
out = audit_c[[c for c in keep if c in audit_c.columns] + rich]
out = out.rename(columns={"model": "predictor"})
out.to_csv(f"{AUDIT_DIR}/audit_long.csv", index=False)
print(out.shape, sorted(out.columns))
```

`audit.io.load_long` accepts either `model` or `predictor` and renames, so the
rename above is belt and braces rather than required. It does raise if any
required column is absent, which is the point of going through it rather than
`pandas.read_csv`.

## Level 3 — from the slides. GPU, token, about fifteen hours.

```bash
# 1. the resource panels, into resources/
#    liana_consensus.csv  mcp_counter.json  progeny_top500.csv
#    gene_universe.json   slide_tissue.csv
#    see resources/README.md for the schema of each

# 2. the prediction cache: one .npz per (slide, predictor),
#    keys Xhat, Xmeas, coords, genes, over the frozen gene universe.
#    docs/datasets.md has the contract and the four invariants that
#    silently change the answer if violated. Ridge, PLS and kNN also
#    need Phikon embeddings first -- see docs/predictors.md
export HF_TOKEN=...

# 3. score every cell
python scripts/run_audit.py --cache $CACHE --out results/audit_long.csv \
       --rich-baseline --error-features

# 4. then level 2
```

On an A100, step 3 is roughly 9 h for the non-domain operators at B=99 and 6 h
for the domain family at B=39. The domain nulls dominate because every
permutation re-runs clustering.

The block permutation null runs here rather than in the analysis stage, since
it needs the predicted field and not the long table:

```bash
python scripts/run_audit.py --cache $CACHE --null block --n-blocks 25 \
       --operators lr_magnitude,lr_hotspot --out results/block_long.csv
```

Check the tiling before committing to a block count. `spatial_nulls.block_report`
reports two limits that are easy to confuse: the permutation floor `1/(1+B)`,
which is a property of B alone, and the number of distinct arrangements `k!`
available from k occupied tiles, beyond which extra draws are repeats. A
four-tile tiling admits 23 arrangements, so B=99 against it spends 76 draws on
duplicates while producing a p-value that looks ordinary.

## Agreement between the notebook and the library

`audit/` is a reimplementation, not the code that was run. The two agree on the
tables in `results/` and the reimplementation is the readable one, but where
they disagree the notebook is what the paper reports. One disagreement is on
record: the non-spline conditioners in the conditioner sweep, documented in
`results/response_period/README.md`.
