# A Multi-Operator Audit of Virtual Spatial Transcriptomics

Code for the two-part audit described in *Does Downstream Biology Add Information
Beyond Gene Accuracy?* Fifteen downstream operators are tested for

- **recoverability** — does the operator detect structure a spatial null does not? and
- **distinctness** — does it explain variance beyond operator-matched gene accuracy?

across 61 HEST-Bench Visium slides, six tissues and six predictors.

Anonymised for review. No author, institution or account identifiers appear in
this repository or its history.

## Layout

```
audit/
  core.py           operator registry, k-NN graphs, permutation nulls, MC p-values, GPU batching
  operators.py      the fifteen operators and their agreement functions
  distinctness.py   D, slide-level cluster bootstrap, paired family contrasts
  baselines.py      operator-matched per-gene accuracy
  analyses.py       reordering, pair-level coupling, dimensionality, tissue, regime
  errors.py         reconstruction-error conditioners
  reliability.py    split-half reliability and the correction for attenuation
  spatial_nulls.py  block permutation, tiling diagnostics
  io.py             long-table schema check and tissue join
  resources.py      LIANA / PROGENy / MCP-counter panels, frozen gene universe
scripts/
  run_audit.py            score every operator on every slide-model pair -> long table
  analyze.py              long table -> operator table, paired deltas, sensitivity
  rebuttal_analyses.py    the analyses added during the response period
  make_synthetic_long.py  a synthetic long table, so the pipeline runs with no data
tests/              invariants the reported numbers depend on
results/            the tables behind every number in the paper
  audit_long.csv    the per-cell scores every other table is computed from
  response_period/  the tables behind the author responses
docs/
  predictors.md         how the six predictors were obtained, and the Phikon prerequisite
  datasets.md           connecting the pipeline to HEST-1k, and the cache contract
  anonymous-release.md  releasing this for double-blind review without deanonymising
```

`REPRODUCING.md` says what this clone can and cannot regenerate on its own,
at four levels of cost. Read it before step 1 of anything.

## Provenance

Two things in this repository have different standing, and conflating them
would be misleading.

The numbers in the paper were produced by a Colab notebook running against a
cached prediction store. That notebook is not released: it is tied to one
Drive layout and carries execution state that is not portable, and releasing
it would add nothing a reader could run.

What takes its place is `results/audit_long.csv`, the per-cell table the
notebook wrote: one row per slide x predictor x operator, carrying the score,
the permutation p, the null SD and the matched per-gene accuracy. Every table
in `results/` is computed from it by `scripts/analyze.py` and
`scripts/rebuttal_analyses.py`, so a reader who wants to check a reported
figure recomputes it from the same per-cell data the paper was built on, in
about twenty minutes and without running a predictor.

`audit/` is a clean reimplementation of the same method, extracted so the
definitions are readable and so the invariants can be tested. Its numbers
agree with the notebook's on the tables in `results/`, but it is not what was
run for the submission. One disagreement between the two is on record, on the
non-spline conditioners in the conditioner sweep; `results/response_period/README.md`
gives it.

Prediction itself is out of scope here. The six predictors were obtained as
described in `docs/predictors.md`, from published checkpoints and from
leave-one-slide-out fits on tile embeddings, and the audit consumes their
cached output. Nothing in this repository trains or runs a predictor.

## Reproducing

`REPRODUCING.md` has the full account. In short, four levels:

```bash
pip install -r requirements.txt

pytest tests/ -q                      # level 0: 56 invariants, no data, <1 min

python scripts/make_synthetic_long.py --out results/synthetic_long.csv
python scripts/analyze.py --long results/synthetic_long.csv --out /tmp/syn \
       --tissue-map results/synthetic_slide_tissue.csv   # level 1: no data, ~10 min
```

Level 0 verifies the method is what the paper describes. Level 1 runs the
analysis half end to end against a known ground truth and recovers Table 1's
three-group membership, though not its values, which the synthetic data does
not and is not meant to reproduce.

Level 2 regenerates the published tables from the per-cell scores, which needs
`results/audit_long.csv`:

```bash
python scripts/analyze.py --long results/audit_long.csv --out results/ \
       --tissue-map resources/slide_tissue.csv
python scripts/rebuttal_analyses.py --long results/audit_long.csv \
       --tissue-map resources/slide_tissue.csv --out results/response_period/
```

Level 3 starts from the slides and needs the resource panels, a HuggingFace
token and the prediction cache. On an A100 it is roughly 9 h for the non-domain
operators at B=99 and 6 h for the domain family at B=39; the domain nulls
dominate because each permutation re-runs clustering.

The test suite pins the properties whose violation silently changes a reported
number rather than raising: the p-value floor at each B, the bootstrap relabel,
the direction of D on inputs whose answer is known, that the domain graph is
destroyed by shuffling coordinates, that each operator's null destroys the
structure that operator measures, that the block permutation preserves
autocorrelation where a spot shuffle does not, and that correcting for
attenuation only ever lowers D.

## Three decisions that change the answer

**Permutation p, not z.** Null widths across these operators span two orders of
magnitude, from σ = 0.003 for hotspot overlap to σ = 0.276 for neighborhood
topology. A fixed z > 3 is a different test on every operator, and it is
systematically stricter where the null is wide (r = 0.76 between null width and
the z-vs-p disagreement). The gate is the fraction of slide-model pairs with
permutation p ≤ 0.05.

**p = (1+b)/(1+B).** The naive b/B admits p = 0, which is not a valid estimate of
a positive probability. The smallest attainable p is therefore 1/(1+B): B = 99
for all operators except the domain family, where each permutation re-runs
clustering and B = 39. Both floors (0.010, 0.025) sit below α = 0.05.

**The domain null re-clusters.** Permuting fitted domain labels is not a null for
a domain operator: it destroys the spatial contiguity that real domains have and
shuffled labels never do, so pure noise certifies at z above 100. Shuffling the
expression matrix and re-running the clustering preserves the contiguity the
graph construction imposes and leaves only biological signal to test; noise then
sits at z = −0.39.

Two further points of construction. The spline basis and ridge penalty are both
fit *inside* each training fold — fitting the basis on the full sample leaks
held-out slides into the representation of accuracy. And slides resampled in the
cluster bootstrap are relabelled, so a slide drawn twice contributes two distinct
cross-validation groups; without the relabel both copies land in the same fold
and the intervals come out too narrow.

## Headline results

Nine of fifteen operators are recoverable. Three of those are near-perfectly
explained by matched gene accuracy; six retain substantial independent variation.

| | operator | B | recoverability [95% CI] | D [95% CI] |
|---|---|---|---|---|
| **A** recoverable and distinct | `spatialdm` | 99 | .596 [.537, .660] | **.863** [.729, .922] |
| | `coexpr` | 99 | .940 [.902, .970] | **.818** [.698, .898] |
| | `domain_boundary` | 39 | .596 [.505, .683] | **.735** [.587, .907] |
| | `lisa` | 99 | .770 [.701, .836] | **.657** [.475, .925] |
| | `lr_hotspot` | 99 | .798 [.740, .844] | **.448** [.338, .556] |
| | `mcp_hotspot` | 99 | .821 [.753, .867] | **.410** [.253, .548] |
| **B** redundant anchor | `lr_magnitude` | 99 | .869 [.822, .915] | .118 [.077, .150] |
| | `progeny` | 99 | .836 [.751, .911] | .099 [.064, .140] |
| | `mcp` | 99 | .917 [.873, .951] | .093 [.056, .136] |
| **C** fails recoverability; D uninterpretable | `svg_top200` | 99 | .262 | 1.000 |
| | `domain_ari` | 39 | .314 | .959 |
| | `domain_nmi` | 39 | .374 | .954 |
| | `moran` | 99 | .344 | .946 |
| | `moran_pea` | 99 | .336 | .934 |
| | `neighborhood` | 99 | .417 | .928 |

The three anchors target unrelated biology — signaling, transcriptional
programs, composition — yet agree to within 0.03. What organizes the result is
functional form: smooth magnitude-based transforms inherit gene accuracy,
thresholded or partitioned transforms do not.

Paired slide bootstrap within matched families, same slides, models and genes,
differing only in agreement function:

| family | ΔD | 95% CI | P(ΔD > 0) |
|---|---|---|---|
| `lr_magnitude` → `lr_hotspot` | **+0.329** | [+0.237, +0.442] | 1.000 |
| `mcp` → `mcp_hotspot` | **+0.317** | [+0.166, +0.450] | 1.000 |
| `moran` → `lisa` | +0.44 | — | not interpretable (magnitude member fails recoverability) |

At pair resolution, over 249,144 ligand-receptor × slide × model observations,
magnitude concordance tracks constituent-gene accuracy at ρ = 0.845 and hotspot
overlap at ρ = 0.602; the paired coupling gap is +0.241 [+0.195, +0.288] and
every predictor reproduces it independently (+0.193 STPath to +0.337 PLS).

Group C is the reason the recoverability gate is load-bearing. Six operators
show D between 0.92 and 1.00 — higher than anything in Group A — while
recovering on 26–42% of slide-model pairs. Their D is high because their scores
are noisy. Reported as distinctness alone, each would look like a newly
discovered evaluation axis.

## Robustness

Everything below is in `results/sensitivity.csv`.

- Conditioner: ΔD for the LR family is +0.31 to +0.35 under spline, linear,
  random forest, gradient boosting and k-NN conditioners.
- Baseline width: replacing the mean matched PCC with eight summary statistics
  of the per-gene accuracy distribution *widens* ΔD to +0.436.
- Leave-one-model-out: all six hold-outs positive, +0.265 to +0.368.
- CV partition: 50 random grouped partitions, all positive.
- Split-half reliability of the LR operators is .994 (magnitude) and .992
  (hotspot); correcting ΔD for measurement noise moves it by 0.001.
- Hotspot quantile: coupling to accuracy stays between 0.67 and 0.80 across
  q = 0.70 to 0.97, against 0.94 for magnitude.
- Recoverability cutoff at {0.3, 0.5, 0.7} yields 11, 6 and 4 distinct-and-
  recoverable operators; the four surviving the strictest cutoff are a subset of
  the six reported.

## Scope

All slides come from one benchmark collection on one platform, so properties of
downstream operators in general cannot be separated from properties of
HEST-processed Visium data. The central mechanism has a structural explanation —
smooth transforms of Ŷ inherit the agreement between Ŷ and Y — that does not
depend on the data source, and the empirical confirmation spans six tissues, six
predictors and three operator families. External replication is still required.

Two operators were excluded before analysis: a Frobenius co-expression variant
and a top-50 gene-overlap variant, both of which produce near-constant nulls.

## Data

HEST-Bench slides are obtained through the HEST release; the loader expects a
HuggingFace token in `HF_TOKEN`. Gene resources are LIANA consensus (ligand-
receptor), PROGENy (pathway weights, cached as a flat table) and MCP-counter
(lineage markers). `resources/README.md` gives the schema of each file.
The panels are third-party and are not redistributed here, so `resources/` is
empty in a fresh clone and only level 3 of `REPRODUCING.md` needs it.

## License

MIT.
