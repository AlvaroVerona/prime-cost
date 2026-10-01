# Prime Cost: engineering log

The decisions that shaped the project and the real problems found while building it. Kept on purpose: finding and fixing them is most of the engineering.

## Decisions

- **A simulator with a hidden truth.** Real restaurant data with known leaks do not exist, so the simulator plants them (over-portioning, over-pouring, unlogged waste,
  expired stock, stock-outs) and records them separately. The analysis never reads that truth; the tests compare against it.
- **One inventory engine, two uses.** The same engine generates the operational data and replays the last 12 weeks under another ordering policy. That makes the
  purchasing comparison fair (same guests, same stock at the start) instead of a formula.
- **Rolling-origin, direct multi-horizon forecasting.** Every (forecast date, target day) pair uses only what was known at the forecast date. Weather gets noise at test
  time so the model is not handed the truth.
- **Common random numbers.** The two strategies in the Monte Carlo, and the two ordering policies in the replay, are run on the same random draws so the difference is the
  action, not luck.
- **Every constant lives in `config/`.** Nothing hardcoded in `src/`.

## Problems found and fixed

1. **Demand censored by stock-outs.** The ordering rule averaged *sold* units, so every stock-out lowered the next order and caused more stock-outs (13% of demand lost).
   The owner knows when a dish sold out, so unserved demand now counts as use. Lost demand fell to 8%.
2. **Guests arriving after the staff had left.** Hour weights sent 4% of weekday guests to midnight while the template ended at 24:00. Fixed in the demand model so the
   staffing comparison is not judged against hours that were never staffed.
3. **A 63% food cost on the best-selling croquettes** came from putting 40 g of ham in six croquettes. Recipe corrected to 20 g (still a Plowhorse, at 39%).
4. **Quarantine cascade biased the reconciliation.** Quarantining the whole ticket whenever one line failed removed about 2.7% of real sales and created a spurious gap between
   theoretical and actual stock use. Records are now quarantined one by one (0.3% of lines).
5. **The forecast interval coverage was trivially 80%.** It was calibrated and measured on the same residuals. It is now calibrated on the first half of the test weeks and
   measured on the second half (84.5%).
6. **Noisy elasticity estimates were being used raw.** A first estimate of -1.8 (true value -0.65, inside the interval) drove the price scenarios. Estimates are now shrunk toward a
   conservative prior in proportion to their uncertainty, and each recommendation is tested against a pessimistic elasticity.
7. **A 19% EBITDA margin was unrealistic.** Added the owner-operator's pay as a cost and revised wages; the margin is now 13.6%. The staffing saving was also being applied to
   total labor although the manager is not optimized; the scenarios now use the saving as a share of total labor (12.5%).
8. **Open wine bottles almost never expired** with a single shelf life. It now depends on the wine type (sparkling 1 day, whites and sherry 2, reds 3).
9. **A wrong assumption in my own test.** I assumed a price rise helps high-margin dishes most; in absolute euros it helps thin-margin dishes more, because the same +5% weighs more
   on a small margin. The test now states the right relationship.
10. **Linear regression does not beat the naive baseline** (20.6% vs 20.3%). Reported as is: the gain of gradient boosting comes from weather-season interactions, not from having
    more variables.
