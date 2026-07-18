# Correction to the originally submitted analysis

During peer review a reviewer asked whether partial-year 2026 data biased the
temporal analysis. Re-examining the code showed that it did.

## What went wrong

`src/04_analysis/temporal_analysis.py` builds

```python
year_range = np.arange(min_year, max_year + 1)   # max_year = 2026
```

and regresses over that whole vector. The originally submitted manuscript
stated that "2026 data excluded due to partial-year indexing", but in fact
every published slope included 2026, which was indexed only through May.

Confirmed by exact reproduction: recomputing with 2026 included reproduces the
published slopes to 0.000000, whereas restricting to complete years changes
them by up to 0.024 %/year.

## What changed after correction

| | As published | Corrected (2005-2025) |
|---|---|---|
| Emerging | 34 | **36** |
| Stable | 33 | **31** |
| Cold | 16 | 16 |

Four topics were reclassified (three Stable to Emerging, one Emerging to
Stable). The rank order of the fastest-growing and fastest-declining topics
was unchanged.

`scripts/revision_analyses/R7_recompute_trends_2005_2025.py` implements the
corrected analysis. Both versions are published for comparison:
`results/temporal_trends_as_published.csv` and
`results/temporal_trends_corrected_2005_2025.csv`.
