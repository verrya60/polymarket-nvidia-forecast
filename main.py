#!/usr/bin/env python3
"""Polymarket "Next CEO of NVIDIA?" forecast (make.fun, program/v1).

Forecasts each market's Yes price 240 minutes ahead as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per minute, scored by pinball loss.

Model (fitted on the six practice cases; practice loss 0.269 vs the house
baseline's 0.364, beating it in every case):

  * Median = the market's last price. No drift: every drift rule tested
    (momentum and mean-reversion, 15/60/240-min windows) made the loss
    worse. The price level is a martingale for practical purposes.
  * Spread per market from its own recent volatility: sigma = 0.65 * sd of
    non-overlapping 5-minute price changes over the last 120 minutes,
    divided by sqrt(5), floored at 0.00030 and capped at 0.005 price units
    per square-root minute. Recent (2h) volatility predicts the next 4h far
    better than 6h/12h/24h windows: the market's regime changes fast.
  * Sticky prices: ~82% of minutes the price does not move at all, so the
    forecast is a mixture -- with probability w(h) the price is still
    exactly at its checkpoint level after h minutes, otherwise it is
    normal with sd sigma*sqrt(h). w(h) = exp(-h/T), T = min(240, 5/f),
    f = share of 5-minute blocks with a price change in the last 180
    minutes. Quantiles are inverted from the mixture CDF, so quiet markets
    get tight inner quantiles and active ones the full normal spread.

Deterministic, standard library only, no network, no randomness.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Volatility window (minutes), scale, floor and cap of sigma (price units
# per square-root minute).
VOL_WINDOW = 120
VOL_SCALE = 0.65
SIGMA_FLOOR = 0.00030
SIGMA_CAP = 0.005
# Activity window (minutes) for the stickiness time constant, and its cap.
ACT_WINDOW = 180
T_CAP = 240.0
# Fallback spread for markets too short to estimate (few prices).
FALLBACK_SIGMA = 0.001


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def market_params(prices):
    """(sigma, T) for one market from its price history (last price last).

    sigma: per-sqrt-minute spread from 5-minute changes over VOL_WINDOW.
    The floor (0.00030) is jump insurance for dead markets: even a market
    that has not moved in hours can jump on news, and the practice says a
    slightly higher floor beats a lower one.
    T: mean minutes between moves, from the share of 5-minute blocks with
    a change over ACT_WINDOW; None when the history is too short to tell
    (then the forecast is a plain normal, no point mass).
    """
    seg = prices[-VOL_WINDOW:] if len(prices) > VOL_WINDOW else prices
    changes = [seg[i + 5] - seg[i] for i in range(0, len(seg) - 5, 5)]
    if len(changes) >= 2:
        sigma = VOL_SCALE * statistics.pstdev(changes) / math.sqrt(5.0)
    else:
        sigma = FALLBACK_SIGMA
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
    if len(changes) < 2:
        return sigma, None
    act = prices[-ACT_WINDOW:] if len(prices) > ACT_WINDOW else prices
    moved = sum(1 for i in range(0, len(act) - 5, 5) if act[i + 5] != act[i])
    blocks = max(1, (len(act) - 5) // 5)
    frac = moved / blocks
    if frac <= 0.0:
        t = 1e9
    else:
        t = 5.0 / frac
    return sigma, min(t, T_CAP)


def forecast_market(prices, horizon, quantiles):
    """240 rows of five quantiles for one market."""
    if not prices:
        # No history at all: a valid, uninformative forecast beats a crash.
        return [[price6(0.5 + norm_ppf(t) * 0.0025 * math.sqrt(h))
                 for t in quantiles] for h in range(1, horizon + 1)]
    x = prices[-1]
    sigma, t_const = market_params(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        if t_const is None:
            row = [x + norm_ppf(t) * spread for t in quantiles]
        else:
            w = math.exp(-h / t_const)
            row = []
            for tau in quantiles:
                if w >= 1.0 - 1e-9:
                    row.append(x)
                    continue
                lo = (1.0 - w) * 0.5
                hi = lo + w
                if tau <= lo:
                    z = norm_ppf(tau / (1.0 - w))
                elif tau >= hi:
                    z = norm_ppf((tau - w) / (1.0 - w))
                else:
                    z = 0.0
                row.append(x + z * spread)
        rows.append([price6(v) for v in row])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        case_id = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = case["horizon_minutes"]
        quantiles = case["quantiles"]
        forecasts = [
            {"id": m["id"],
             "q": forecast_market(m["prices"], horizon, quantiles)}
            for m in case["markets"]
        ]
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts},
                                separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
