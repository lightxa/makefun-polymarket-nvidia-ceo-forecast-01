#!/usr/bin/env python3
"""Polymarket NVIDIA CEO forecaster v2.

Median is persistence (last price). The spread is a per-market robust
volatility blended from two lags, because 1-minute midpoint diffs are
mostly tick jitter while 3-minute changes carry the real short-horizon
moves:

  sigma = 0.5 * sig1 + 0.5 * sig3

where sig1 is a winsorized blend of RMS and mean-absolute 1-minute
changes over the last 360 minutes, and sig3 is the same estimator on
3-minute changes over the last 1440 minutes, divided by sqrt(3).
Both estimators winsorize at the 99.5th (sig1) / 95th (sig3) percentile
of absolute changes, so the early-history spike chaos in these markets
does not inflate the spread. The spread then grows sub-diffusively,

  q_tau(h) = x + z_tau * 1.2 * sigma * h**0.30,

because these prices are steppy and mean reverting, not a random walk:
a sqrt(h) spread (the house baseline) is far too wide by h=240, and
alpha near 0.30 with scale 1.2 sits in the middle of a flat backtest
plateau (alpha 0.26-0.34, scale 1.1-1.3 all score within ~0.005).

Robustness: markets younger than the lag window fall back to sig1
alone; sigma is floored at 0.0002. Standard library only, deterministic,
offline, no network, no randomness.
"""
import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
WINDOW_1MIN = 360
WINDOW_LAG3 = 1440
ALPHA = 0.30
SCALE = 1.2
SIGMA_FLOOR = 0.0002
SIGMA_CAP = 0.012


def _winsor_scale(abs_changes, cap_pct):
    """Robust per-step sigma from sorted absolute changes.

    0.6 * winsorized RMS + 0.4 * mean-absolute / E|N(0,1)|.
    """
    n = len(abs_changes)
    if n == 0:
        return None
    cap = abs_changes[min(n - 1, int(cap_pct * n))]
    if cap <= 0:
        cap = abs_changes[-1] if abs_changes[-1] > 0 else 0.0005
    rms = math.sqrt(sum(min(d, cap) ** 2 for d in abs_changes) / n)
    meanabs = sum(abs_changes) / n
    return 0.6 * rms + 0.4 * (meanabs / 0.7979)


def sigma_1min(prices):
    seg = prices[-WINDOW_1MIN:] if len(prices) > WINDOW_1MIN else prices
    if len(seg) < 3:
        return None
    diffs = sorted(abs(seg[i] - seg[i - 1]) for i in range(1, len(seg)))
    return _winsor_scale(diffs, 0.995)


def sigma_lag3(prices):
    seg = prices[-WINDOW_LAG3:] if len(prices) > WINDOW_LAG3 else prices
    lag = 3
    if len(seg) <= lag + 2:
        return None
    ch = sorted(abs(seg[t] - seg[t - lag]) for t in range(lag, len(seg)))
    sig_block = _winsor_scale(ch, 0.95)
    if sig_block is None:
        return None
    return sig_block / math.sqrt(lag)


def market_sigma(prices):
    s1 = sigma_1min(prices)
    s3 = sigma_lag3(prices)
    if s1 is None and s3 is None:
        return 0.001
    if s1 is None:
        sig = s3
    elif s3 is None:
        sig = s1
    else:
        sig = 0.5 * s1 + 0.5 * s3
    return max(SIGMA_FLOOR, min(SIGMA_CAP, sig * SCALE))


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon):
    x = prices[-1]
    sig = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        s = sig * (h ** ALPHA)
        rows.append([price6(x + z * s) for z in Z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[:-3]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        if case["quantiles"] != QUANTILES:
            raise ValueError(f"{fname}: unexpected quantiles {case['quantiles']}")
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for m in case["markets"]:
            if not m["prices"]:
                raise ValueError(f"market {m['id']} has no prices")
            forecasts.append({"id": m["id"], "q": forecast_market(m["prices"], horizon)})
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
