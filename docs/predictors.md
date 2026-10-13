# Predictors

Six predictors produce Ŷ for every slide. They mix information regimes
deliberately: the audit is about operators, not architectures, and a spread of
reconstruction quality is what makes the conditioning on gene accuracy
informative. Rankings are reported descriptively and support no architectural
claim.

| predictor | regime | source |
|---|---|---|
| STPath | pretrained transfer | public checkpoint, no fine-tuning |
| DeepSpot-M | pretrained transfer | public checkpoint, no fine-tuning |
| OmiCLIP | in-domain retrieval | nearest reference slide among the other 60 |
| Ridge | in-domain, leave-one-slide-out | fit on Phikon tile embeddings |
| PLS | in-domain, leave-one-slide-out | `n_components = 8`, same embeddings |
| kNN | in-domain, leave-one-slide-out | `k = 15`, same embeddings |

The three in-domain predictors are not image models. They are linear or
instance-based regressions on top of a frozen histology encoder, and that
encoder is a prerequisite you have to obtain before any of the three can be
reproduced. The next section says which one and why it matters.

Count-scale outputs are mapped to log(1+x) before scoring, as are measured
counts, so both sides of every comparison share a scale.

## The tile encoder the linear predictors depend on

Ridge, PLS and kNN each consume a frozen per-tile embedding and never see
pixels. Reproducing them therefore has a prerequisite the other three
predictors do not: **Phikon**, a self-supervised histology encoder.

> Filiot A, Ghermi R, Olivier A, Jacob P, Fidon L, Mac Kain A, Saillard C,
> Schiratti J-B. Scaling self-supervised learning for histopathology with
> masked image modeling. *medRxiv* 2023. doi:10.1101/2023.07.21.23292757.
> Preprint.

| | Phikon | Phikon-v2 |
|---|---|---|
| HuggingFace id | `owkin/phikon` | `owkin/phikon-v2` |
| architecture | ViT-Base/16, ~86M params | ViT-Large/16, ~307M params |
| pretraining | iBOT, masked image modelling plus self-distillation | DINOv2 |
| pretraining data | PanCancer40M: >40M tiles from ~6k TCGA slides, 16 cancer types | ~456M tiles from 58,359 WSIs across TCGA, CPTAC, GTEx |
| **embedding width** | **768** | **1024** |
| licence | Owkin, non-commercial | Owkin, non-commercial |

```python
from transformers import ViTModel
enc = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
# per tile: take the CLS token, giving a 768-vector
emb = enc(pixel_values=tiles).last_hidden_state[:, 0, :]
```

**Three things that will silently change your numbers.**

*The two versions are not interchangeable.* 768 against 1024 dimensions is a
different regression problem. If you reproduce these predictors with
Phikon-v2, say so, because the result is a different predictor with the same
name. [CONFIRM which version produced the cached predictions before releasing
this file.]

*Reduce before you regress.* HEST-Bench applies PCA to 256 components before
Ridge, specifically so that encoders with different embedding widths are not
compared on an unequal footing — a wider embedding otherwise gives Ridge more
parameters and flatters it. We follow that, so the Ridge, PLS and kNN
predictors here are fit on 256 PCA components of the Phikon embeddings rather
than on the raw 768.

*The licence is non-commercial.* Phikon weights cannot be redistributed inside
a repository. Pull them from HuggingFace at build time.

Nothing about the audit depends on Phikon being the right choice of encoder.
These three exist to widen the range of reconstruction accuracy the conditioner
observes, which is what makes conditioning informative. A different encoder
would shift their accuracy and change nothing about the operator comparison,
because every operator is conditioned on each predictor's own realised accuracy
rather than on a cross-predictor ranking.

## Frozen gene universe

Predictions are cached once per (slide, predictor) over a frozen universe of
6,970 symbols: the union of the ligand-receptor, pathway, marker and
highly-variable gene sets, intersected with the vocabularies all six predictors
emit. Every operator is therefore scored on identical predictions, and no
operator can be advantaged by a predictor emitting a gene another does not.

Rebuilding the universe invalidates the cache. `resources.gene_index` keys on
the full gene tuple rather than a length-and-endpoint digest: marker panels
frequently share a length and both endpoints, and a digest key silently returns
another panel's column indices.

## PLS component count

PLS at `n_components = 8` runs in about three minutes per slide. Larger
component counts were tested and cost roughly 11 minutes per slide for no change
in the audit: the operator table and both paired contrasts are unaffected,
because the audit conditions on each predictor's own realised gene accuracy
rather than comparing predictors to each other.

## Per-predictor reproduction of the mechanism

The magnitude-to-localization coupling gap is present in every predictor
independently:

| predictor | coupling gap |
|---|---|
| STPath | +0.193 |
| DeepSpot-M | +0.214 |
| OmiCLIP | +0.229 |
| kNN | +0.251 |
| Ridge | +0.288 |
| PLS | +0.337 |

Pooled paired bootstrap: +0.241 [+0.195, +0.288]. This rules out the effect
being a property of any one predictor's error structure.

Under matched gene accuracy the descriptive ordering is PLS .175, Ridge .162,
STPath .151, kNN .142, DeepSpot-M .140, OmiCLIP .055. Residual analysis finds no
predictor systematically exceeding its own accuracy on any operator, with mean
residuals within ±0.01 except on the two operators with the widest nulls.
