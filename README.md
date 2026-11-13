---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

<!-- dataset-factory:links -->
**Dataset Factory · v-498e55482c3ed94b**

[Dataset website](https://earino.github.io/factory-eia861-reliability-deterioration/versions/v-498e55482c3ed94b/) · [GitHub release](https://github.com/earino/factory-eia861-reliability-deterioration/releases/tag/v-498e55482c3ed94b) · [Hugging Face](https://huggingface.co/datasets/earino/factory-eia861-reliability-deterioration/tree/v-498e55482c3ed94b) · [Catalogue](https://earino.github.io/dataset-factory-site/)

Artifact: 6aa939acba34270679973c9e8b5ea12984814b146a32ca50ec71d3b23ac2790a
<!-- /dataset-factory:links -->

# Forward prediction of day-to-day distribution reliability deterioration at U.S. electric utilities from Form EIA-861 reliability history

Given three years of published distribution-reliability history and utility attributes, will a U.S. electric distribution utility's day-to-day outage duration (SAIDI excluding major event days) jump by at least 50% above its own recent norm in the next calendar year?

Derived from U.S. Energy Information Administration (EIA), U.S. Department of Energy. This is not its official product.

Prediction time: 31 December of forecast year Y, using only Form EIA-861 reliability and utility attribute filings for calendar years Y-3, Y-2 and Y-1. Exact historical publication dates are assumed rather than verified. No CY Y or later value enters features.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.6963780731791774,
    "brier": 0.1190485278420198
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.12500660216769513
  },
  "calendar_identity": {
    "auc": 0.6172737678427147,
    "brier": 0.15601477427071603
  }
}
```

## Limitations

- Reliability indices are self-reported and unaudited by EIA; utilities may use IEEE 1366 or an internally defined method, and non-IEEE reporters self-determine which days are major event days, so the excluding-MED values are not strictly comparable across utilities.
- Annual reliability is dominated by weather and storm occurrence, and the feature set deliberately contains no storm, outage-cause or capital-spending covariates, so a large share of label variance is unpredictable by construction; the task measures screening skill, not causal drivers.
- This is a current-vintage retrospective built from today's published files: the values a regulator would have seen at the decision instant could differ from the final vintage, so no claim of proven historical operational availability is made.
- Workbook layout is not stable across vintages: IEEE and other-standard column blocks coexist, some early files are .xls, and index columns were reportedly relabelled around 2021, so the parser must match headers by pattern and be re-checked for each new year.
- The panel recurs utilities across cohorts; features are capped at CY Y-1 and a purge is enforced at split boundaries, but autocorrelation within splits still makes the effective sample size smaller than the row count.
- Utilities that merge, change EIA utility number, file multiple state rows, or switch reporting standard are dropped, which removes some large multi-state investor-owned utilities and slightly favours stable filers.
- The label multiplies a utility's own baseline by a fixed 1.5 factor, so a utility with an unusually low or unusually volatile three-year baseline is mechanically easier or harder to flag; the rule is frozen before any performance measurement and not re-tuned.
- The assumption that a calendar year's final EIA-861 reliability data are public by 31 December of the following year is inferred from observed release timing and the publisher's file listing; freeze exact file vintages and record checksums in a manifest so a later re-release cannot silently change the benchmark.
- EIA also publishes early-release versions that exclude utilities pending validation; only final-vintage ZIPs should be used, and the 2025 file should be confirmed final at build time.
- Upstream third-party material is excluded: EIA photographs are privately licensed and the EIA logo is a registered trademark that must not be reproduced; only the numeric data are redistributed, and EIA's requested acknowledgment with a publication date must accompany any release.
- Reliability indices are self-reported and unaudited by EIA. Utilities may use IEEE 1366 or an internally defined method, and non-IEEE reporters self-determine major event days, so excluding-MED values are not strictly comparable across utilities.
- Annual reliability is dominated by weather and storm occurrence. The declared feature set contains no storm, outage-cause or capital-spending covariates, so a large share of label variance is unpredictable by construction.
- This is a current-vintage retrospective built from today published files. Values a regulator would have seen at the decision instant could differ from the final vintage, and exact historical publication dates were not verified.
- Workbook layout is not stable across vintages. The parser matches headers by text pattern, allows both IEEE and other-standard blocks, and fails with observed headers when required fields are missing. Legacy .xls vintages are read by an embedded BIFF reader rather than an external xls package.
- The panel recurs utilities across cohorts. Features are capped at CY Y-1 and no holdout is used to choose the target, features, thresholds or splits, but autocorrelation within splits still makes effective sample size smaller than row count.
- Utilities that merge, change EIA utility number, file multiple state rows or switch reporting standard are dropped. This removes some large multi-state investor-owned utilities and slightly favors stable filers.
- The label multiplies a utility own baseline by a fixed 1.5 factor, so a utility with an unusually low or volatile three-year baseline is mechanically easier or harder to flag. The rule was frozen before measurement and is not re-tuned.
- The assumption that a calendar year final EIA-861 reliability file is public by 31 December of the following year is inferred from release practice, not from a complete publication ledger. Timing observations are deliberately empty because actual publication dates were not verified; the 365-day value is a conservative calendar bound, not a measured publisher lag.
- No persistence score column is shipped. The lag-1 SAIDI level is a feature, not a calibrated persistence score for a relative jump, so the persistence reference is null rather than fabricated.
- The 2025 file is treated as final based on publisher evidence. If a later re-release changes it, the holdout labels can move.
- The annual source has no day-level event date. Decision dates collapse to one day per forecast year, so event clustering uses the utility identity as the conservative independent unit available; this is coarser than a genuine outage-event date.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
