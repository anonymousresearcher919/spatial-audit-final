# Connecting the pipeline to real data

The audit consumes a prediction cache: one array of predicted expression and
one of measured expression per slide and predictor, over a frozen gene
universe. Building that cache is the only step that touches a dataset. This
document covers where the slides come from, how the gating works, and what has
to be true of the cache before `scripts/run_audit.py` will produce comparable
numbers.

## HEST-1k

Every slide in the paper comes from HEST-1k.

> Jaume G, Doucet P, Song AH, Lu MY, Almagro-Pérez C, Wagner SJ, Vaidya AJ,
> Chen RJ, Williamson DFK, Kim A, Mahmood F. HEST-1k: A Dataset for Spatial
> Transcriptomics and Histology Image Analysis. *NeurIPS 2024 Datasets and
> Benchmarks Track* (Spotlight). arXiv:2406.16192.

Two HuggingFace repositories matter:

| repository | what it is |
|---|---|
| `MahmoodLab/hest` | the dataset itself, over 1 TB, CC BY-NC-SA 4.0 |
| `MahmoodLab/hest-bench` | the benchmark splits and task definitions |

**The sample count depends on which version you pull.** The arXiv abstract says
1,108, the HuggingFace card currently says 1,229, and the GitHub README says
1,276. The dataset grows. Record the version you downloaded, because an audit
run against a later release is not comparing against the same population. Our
61 Visium slides were selected from the release current at the time, and
`resources/slide_tissue.csv` pins exactly which ones by identifier.

### Access

The repository is public but gated: you accept the conditions on the
HuggingFace page, which means agreeing to share your contact information, and
access is granted automatically. After that, downloads need a token.

```bash
export HF_TOKEN=hf_...            # from huggingface.co/settings/tokens
huggingface-cli login --token $HF_TOKEN
```

This is lighter gating than the UNI and UNI2 encoders, which need manual review
and an institutional email. If you plan to compare against those, start the
request early; it is not instant.

### Selecting the cohort

Our 61 slides are the human Visium samples in six tumour types: 24 clear-cell
renal carcinoma, 23 prostate adenocarcinoma, four colon adenocarcinoma, four
rectal adenocarcinoma, four lymph-node invasive ductal carcinoma, two
hepatocellular carcinoma. Every slide is a tumour, which is a stated scope
limit rather than an accident.

Two filters are worth applying explicitly, because both change the answer and
neither is obvious:

- **Species.** HEST-1k includes mouse samples. Of 602 Visium samples, 421 are
  human. A survey that forgets this overstates the available pool by about
  thirty per cent.
- **Patient-level independence.** Sample count is not slide count is not
  patient count. The largest block of colorectal samples outside our cohort is
  62, of which 41 come from a single patient in one study. Treated as 62
  independent slides they would badly inflate any interval, since the cluster
  bootstrap in `audit/distinctness.py` resamples slides and assumes slides are
  the independent unit.

## HEST-Bench's own protocol, and where we depart from it

HEST-Bench evaluates gene-expression prediction across nine organs and eight
cancer types. Its protocol:

| element | HEST-Bench |
|---|---|
| input | 112 × 112 µm patches at 0.5 µm/px, so 224 px |
| target | log1p-normalised expression of each task's top 50 highly variable genes |
| splits | patient-stratified cross-validation |
| headline model | PCA to 256 components on frozen embeddings, then Ridge |
| metric | Pearson correlation, averaged across genes and folds |

We depart from it in one way that matters. HEST-Bench scores 50 genes per task;
the audit needs every gene that feeds any of the fifteen operators, so our
frozen universe is 6,970 symbols — the union of the ligand-receptor, pathway,
marker and highly-variable sets, intersected with what all six predictors can
emit. A cache built to the HEST-Bench 50-gene target will not run this audit.

The universe is frozen deliberately. Every operator is scored on identical
predictions, so no operator can be advantaged by a predictor emitting a gene
another does not. Rebuilding it invalidates the cache.

## The cache contract

`scripts/run_audit.py` expects one `.npz` per slide and predictor, named
`{slide}__{predictor}.npz`, with four keys:

| key | shape | meaning |
|---|---|---|
| `Xhat` | (n_spots, n_genes) | predicted expression, log1p scale |
| `Xmeas` | (n_spots, n_genes) | measured expression, log1p scale |
| `coords` | (n_spots, 2) | spot coordinates in pixels or µm, consistent within a slide |
| `genes` | (n_genes,) | HGNC symbols, column-aligned to both matrices and identical across every file |

Four invariants the audit depends on and does not check for you:

1. **Both matrices are log1p.** Count-scale predictions compared against
   log-scale measurements produce an operator score that is mostly a scale
   artifact. `audit/` assumes the mapping has already happened.
2. **`genes` is identical across every file.** Not merely the same length: the
   same tuple in the same order. `resources.gene_index` keys on the full tuple
   rather than a length-and-endpoint digest, because marker panels frequently
   share a length and both endpoints and a digest silently returns another
   panel's columns.
3. **`coords` are physical, not array indices.** The k-NN graph and the block
   permutation both read geometry. Row or column indices produce a graph that
   looks plausible and is wrong.
4. **Spot order matches between `Xhat` and `Xmeas`.** Nothing downstream can
   detect a permuted row order; it will simply report that the predictor is bad.

## Other spatial transcriptomics datasets

Three datasets a reader may expect, and the specific reason each is not a
drop-in:

- **spatialLIBD / DLPFC** is healthy dorsolateral prefrontal cortex. It would
  test platform and tissue generalisation at the cost of the oncology scope the
  rest of the work is stated in.
- **HER2ST** uses the legacy ST platform: roughly 100 µm spots, about a
  thousand spots per section. A result there confounds platform with spatial
  resolution, and the block permutation in particular needs enough spots per
  tile to be meaningful.
- **STImage-1K4M** mixes platforms and species, so it needs per-subset
  harmonisation before any of it is comparable.

All three also need their own preprocessing and gene-harmonisation stack, and
two of the six predictors ship as fixed pipelines keyed to the HEST-Bench
format, so porting them changes the predictors rather than adding a cohort.
