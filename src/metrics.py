"""
metrics.py

One job: take a plain list of numbers (say, ten generation-speed measurements)
and describe their SHAPE. Not just the average -- also how spread out they are.
Reporting the spread is the honest part; a lone average can hide a lot.
"""

import statistics


def summarize(values):
    """Return count, mean, median, min, max, and standard deviation."""
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0,
                "min": 0.0, "max": 0.0, "stdev": 0.0}

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),          # the middle value (a.k.a. p50)
        "min": min(values),
        "max": max(values),
        # stdev needs at least 2 numbers; with 1 there's no spread to report.
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def percentile(values, p):
    """Linear-interpolated percentile. p is 0-100 (e.g. 95 for P95)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
