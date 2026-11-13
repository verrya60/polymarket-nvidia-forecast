# Polymarket NVIDIA CEO forecast — entry

Forecasts the Yes price of every market of Polymarket's event
"Next CEO of NVIDIA?" 240 minutes ahead, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per minute, scored by mean pinball loss.
Program: `main.py` (Python 3.13, standard library only, no network,
deterministic).

## Model

- **Median = the market's last price.** No drift. Every drift rule tested
  (momentum and mean-reversion on 15/60/240-minute windows) made the
  practice loss worse — the price level is a martingale for practical
  purposes.
- **Spread per market from its own recent volatility.** sigma = 0.65 × sd
  of non-overlapping 5-minute price changes over the last 120 minutes,
  divided by √5, floored at 0.00030 and capped at 0.005 price units per
  square-root minute. A 2-hour window predicts the next 4 hours far better
  than 6h/12h/24h windows: the market's volatility regime changes fast.
  (The house baseline uses one fixed 0.0025 for every market.)
- **Sticky prices.** ~82% of minutes the price does not move at all, so the
  forecast is a mixture: with probability w(h) = exp(−h/T) the price is
  still exactly at its checkpoint level after h minutes, otherwise it is
  normal with sd sigma·√h. T = min(240, 5/f), where f is the share of
  5-minute blocks with a price change in the last 180 minutes. Quantiles
  are inverted from the mixture CDF, so quiet markets get tight inner
  quantiles and active ones the full normal spread.
- Quantiles are clamped to [0, 1] and rounded to 6 decimals.

## Practice scores

Scored locally against Polymarket's own price history (the kit's six
practice cases p00–p05, answers to 2026-11-13T06:00:00Z). My scorer was
validated by reproducing the house baseline's published practice loss
exactly (0.363589).

| case | this entry | house baseline |
|------|-----------|----------------|
| p00  | 0.481138  | 0.536138 |
| p01  | 0.321132  | 0.380352 |
| p02  | 0.343867  | 0.408408 |
| p03  | 0.233044  | 0.368599 |
| p04  | 0.195297  | 0.298794 |
| p05  | 0.108795  | 0.238232 |
| **overall** | **0.269022** | **0.363589** |

26% lower pinball loss than the baseline, beating it in every case.
90% coverage 0.76–0.95 (baseline 0.87–1.00): the narrower quantiles trade
some coverage for lower pinball loss, which is the ranking metric.

## What did not work (tested and dropped)

- Smoothed anchors (EWMA/mean of the last 15–60 min): much worse
  (0.43–0.51) — the price level moves too fast to lag it.
- Any drift (momentum or reversion): worse at every window and strength.
- Renormalising medians so the Yes prices sum to 1: worse (the universe
  was incomplete at early checkpoints).
- Heavier tails (Student-t quantiles, wider outer quantiles): worse.
- Longer volatility windows (6h/12h/24h) or hourly-change volatility:
  worse — recency wins.
- Constant sigma for all markets: 0.327 at best, vs 0.269 per-market.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory (one case per file) and writes
one `<id>.out` per case. Runs in ~0.1 s per case on one CPU, outputs are
185–246 KB (limit 1 MiB), and two runs produce byte-identical files.

## Limits

- The practice is only six cases over ~24 h; the final is twelve cases
  over 48 h of unseen prices. The model adapts per case (it only uses
  that case's own 72 h history), but a news event in the final window
  (e.g. an NVIDIA succession announcement) would move prices further
  than any quantile here covers — no forecast can predict that.
- The tuning surface is flat near the chosen parameters (floor
  0.0002–0.0003, scale 0.6–0.7 all score within ~0.002), so the exact
  constants are not critical; the structure (last price + recent
  per-market volatility + stickiness) is what carries the gain.
- The house model forecast (an AI's per-market sigma + drift, made once
  for the final) is the other bar to beat; its practice loss is not
  published, so the comparison is against the baseline only.
