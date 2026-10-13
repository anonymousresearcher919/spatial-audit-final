# Resources

Place the frozen resource copies here. `audit/resources.py` reads from this
directory, overridable with the `AUDIT_RESOURCES` environment variable.

| file | contents |
|---|---|
| `liana_consensus.csv` | columns `ligand,receptor`; LIANA consensus, complexes dropped |
| `mcp_counter.json` | `{lineage: [marker symbols]}` for eight of the ten MCP-counter populations |
| `progeny_top500.csv` | columns `pathway,gene,weight`; PROGENy top-500, flattened so the audit has no runtime dependency on decoupler |
| `gene_universe.json` | the 6,970-symbol frozen universe; rebuilding it invalidates the prediction cache |
| `slide_tissue.csv` | columns `slide,tissue` for the per-tissue contrast |

## Why eight MCP-counter populations and not ten

MCP-counter defines ten populations: eight immune (T cells, CD8+ T cells, NK
cells, cytotoxic lymphocytes, B lineage, monocytic lineage, myeloid dendritic
cells, neutrophils) and two stromal (endothelial cells, fibroblasts).

`audit/operators.py` scores eight of them, dropping CD8+ T cells and cytotoxic
lymphocytes. Both are nested inside panels already scored: CD8+ T cells inside
T cells, cytotoxic lymphocytes overlapping T and NK cells. The operator-matched
accuracy baseline takes the union of marker genes feeding an operator, and
nested panels would weight their shared markers twice in that union, which is
the one place in the audit where the panel has to be a set rather than a
multiset.

`mcp_counter.json` must contain every key in `operators.LINEAGE_ORDER`, spelled
exactly as MCP-counter names them. `baselines.matched_genes` raises with the
missing keys listed if it does not.
