# Weekly expanding-window evaluation

Pre-policy chronological 80/20 split; weekly origins; actual future international product prices are supplied. Scores are conditional supply-price estimates, not unconditional crude-oil forecasts.

Test weeks: 2023-12-16 to 2026-03-07. Lead-specific sample sizes: 117, 114, 105, 92, 75.

| fuel | weeks ahead | baseline MAE | recent-margin MAE | TimesFM MAE | no-change MAE | TimesFM improvement vs recent margin (%) |
|---|---:|---:|---:|---:|---:|---:|
| diesel | 1 | 17.55 | 11.30 | 10.29 | 16.96 | 8.9 |
| diesel | 4 | 17.83 | 12.06 | 13.19 | 43.62 | -9.4 |
| diesel | 13 | 19.00 | 13.77 | 14.55 | 58.36 | -5.6 |
| diesel | 26 | 19.88 | 14.10 | 14.26 | 62.59 | -1.1 |
| diesel | 43 | 22.09 | 17.16 | 12.48 | 53.86 | 27.3 |
| gasoline | 1 | 16.89 | 12.00 | 8.49 | 13.59 | 29.3 |
| gasoline | 4 | 17.37 | 12.65 | 11.63 | 32.16 | 8.1 |
| gasoline | 13 | 17.88 | 13.83 | 11.11 | 49.37 | 19.7 |
| gasoline | 26 | 17.14 | 15.06 | 10.94 | 49.60 | 27.4 |
| gasoline | 43 | 19.04 | 19.57 | 11.53 | 42.70 | 41.1 |
| kerosene | 1 | 17.78 | 18.25 | 18.05 | 24.60 | 1.1 |
| kerosene | 4 | 18.04 | 18.75 | 17.81 | 38.31 | 5.1 |
| kerosene | 13 | 18.37 | 19.38 | 20.57 | 53.83 | -6.1 |
| kerosene | 26 | 17.77 | 18.17 | 20.51 | 62.87 | -12.9 |
| kerosene | 43 | 18.95 | 19.31 | 20.94 | 54.20 | -8.5 |

Gasoline: TimesFM has the lowest MAE at all five leads.
Diesel: TimesFM wins at 1 and 43 weeks; recent-margin regression wins at 4, 13 and 26 weeks.
Kerosene: baseline regression wins at 1, 13, 26 and 43 weeks; TimesFM wins at 4 weeks.
MAE and RMSE rankings may differ. For diesel at 1 week TimesFM has lower MAE but higher RMSE than recent-margin regression.
Different leads cover different target-date subsets, so changes across lead scores are not pure horizon effects.
The test period was previously inspected; these comparisons are descriptive and not a new untouched test.

Verification: source hash, identical regression origins/targets/predictions, endpoint alignment, split boundaries and unique target weeks per lead passed.
