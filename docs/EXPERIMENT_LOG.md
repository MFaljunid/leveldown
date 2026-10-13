# leveldown — Experiment Log

Chronological record of every experiment, its configuration, raw output, and the decision it led to.
Terminal outputs are copied **verbatim**. Results are never overwritten (one folder per run).

---

## L0 — Background (from fair-fed-ncf docs/PERSONALIZATION_LOG.md, 2026-10-11)

Study: leveling down in user-fair recommendation.
- C1 (audit): user-fairness interventions narrow group gaps mainly by LOWERING the better-off group.
  Evidence so far: q-FFL on FCF (ML-1M, full ranking, 2 seeds: best group −0.014/−0.020, worst group
  +0.002/+0.004 within seed spread), fair-elicit E47, fair-prior P0, privfair P1, MGDA pre-test.
- C2 (profile difficulty): REJECTED on ML-1M — profile diversity and mainstreamness explain ≈ 3% of
  the activity gap.
- C2' (target difficulty), EXPLORATORY on ML-1M (FCF, full ranking): active users' held-out items are
  much less popular (27% vs 54% in the top-10% popular); test-item popularity explains ≈ 54% of the
  most- vs least-active NDCG@10 gap (OLS) and matching on it shrinks the gap by ≈ 83%
  (−0.0475 → −0.0082). Not testable on LastFM-2K (no activity spread) or Amazon-Electronics
  (backbone below ItemPop).
- Decision: continue in a centralized setting with strong standard backbones.

---

## L1 — C2' pre-registered confirmation on ML-100K (recorded 2026-10-11, BEFORE running on real data)

Data: MovieLens-100K (users from 1997–98, independent of ML-1M), every rating an implicit positive,
temporal leave-one-out (last interaction = test, second last = validation), users with ≥ 5 ratings.
Evaluation: FULL RANKING of the held-out item, known positives excluded, HR@10 / NDCG@10.
Models: iALS (primary; grid dim 64 × alpha {1, 5, 20} × reg {1, 10, 100}, 10 iterations), EASE
(secondary; lambda {50, 200, 500, 2000}), ItemPop (reference); selection on validation NDCG@10.
Groups: activity quartiles of #train items (equal-sized, 0 = least active).
Covariate: log(1 + train popularity) of the test item.
Analysis: gap between the most- and least-accurate activity group (per model); OLS share explained
with bootstrap 95% CI (1000); matching on test-popularity deciles.

**Rule (primary = iALS, NDCG@10): C2' CONFIRMED if (i) iALS beats ItemPop, (ii) the share explained
is ≥ 40%, and (iii) |matched gap| ≤ 50% of |raw gap|.** EASE reported as a robustness check.
If not confirmed: C2' is reported as ML-1M-specific; C1 is assessed on its own.

Code check (synthetic data with popularity-biased consumption, 943 users × 1682 items): pipeline runs
end-to-end in ≈ 30 s; the synthetic data also show the mechanism (share ≈ 66%), indicating that the
effect can arise mechanically from popularity-biased consumption under leave-one-out — to be
discussed as part of the explanation, not as evidence.

---
