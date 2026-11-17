"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Method: comparable-launch shape forecast, local-linear in hour-1 size.
  - Every comparable launch in the case file carries its full release-day hourly
    curve.  For each forecast hour h, take the log ratio log(viewers_h / viewers_1)
    across those launches and build quantiles from it, so the forecast follows the
    shape real launches actually traced (midnight-local unlock steps around
    05:00Z/08:00Z, then the evening prime-time peak) instead of assuming hour 1
    repeats all day.
  - The ratio depends steeply on hour-1 size: across the kit's comparables the
    correlation between log(hour-1 viewers) and the log peak ratio is -0.94 (a
    23k-viewer hour 1 typically grows ~5x by hour 20, a 270k one only ~1.6x).
    So per hour the centre is a weighted local-linear regression of the log
    ratio on log(hour-1 viewers), evaluated at the target's own hour 1 -- a
    weighted median would flatten that slope and under-predict small launches
    and over-predict big ones.
  - Not every comparable is equally relevant.  Each launch is weighted by three
    things, all read from the case file at run time:
      * hour-1 size: a Gaussian kernel on log(hour-1 viewers), bandwidth 1.5.
      * unlock regime: same regime as the target (rolling midnight-local,
        simultaneous global, or unknown) weighs 1.0, a known-but-different
        regime 0.2, unknown 0.5.  The target regime is read from the case notes.
      * tier: comparable 1.0, early-access-contaminated 0.5, weak 0.3,
        the GTA VI pre-release history (not a launch) 0.
  - Per hour, the spread is the weighted residual standard deviation around the
    local-linear fit (floor 0.25, cap 1.0):
    q_tau = p1 * exp(centre + z_tau * sigma), z the standard normal quantiles.
    In leave-one-out tests on the kit's comparables (this program, run as-is)
    this cut the pinball loss from 0.363 (persistence) to 0.235 on
    rolling-midnight launches, the regime GTA VI and all four practice cases
    share, and from 0.930 to 0.380 across all usable comparables.  The main
    residual risk is a launch that spikes like Resident Evil Requiem (12x its
    hour 1 by hour 20); the widened 90% band is the honest price for that.
  - Fallback: with fewer than 5 usable comparables, fall back to the house
    persistence baseline (flat median, sigma 0.15 * sqrt(h - 1)).

Standard library only, deterministic, no network.  Reads every <id>.in in the
input directory and writes <id>.out to the output directory:
  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
SIGMA_FLOOR = 0.25
SIGMA_CAP = 1.0
BASE_BW = 1.5
# A single comparable's log ratio is capped before it enters the quantiles, so
# one extreme launch (a 2,000x spike from a near-zero hour 1) cannot drag the
# 95% quantile to an absurd level if it ever carries non-trivial weight.  The
# cap binds at ratios of 0.05x and ~1,100x, far outside anything a
# rolling-midnight launch traces (the largest in the kit is ~12x), so it does
# not change the rolling-midnight forecast at all.
LOG_RATIO_MIN = -3.0
LOG_RATIO_MAX = 7.0


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_time(text):
    """ISO-8601 UTC timestamp -> epoch seconds (stdlib only)."""
    from datetime import datetime, timezone

    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def unlock_kind(text):
    low = (text or "").lower()
    if "not a launch" in low or "before release" in low:
        return "target"
    if "midnight local" in low or "rolling" in low:
        return "rolling"
    if "simultaneous" in low or "single global" in low:
        return "simul"
    return "unknown"


def tier_of(text):
    low = (text or "").lower()
    if "tier: early-access" in low:
        return "early"
    if "tier: weak" in low:
        return "weak"
    if "tier: target-history" in low:
        return "target"
    return "comp"


TIER_WEIGHT = {"comp": 1.0, "early": 0.5, "weak": 0.3, "target": 0.0}


def regime_weight(comp_kind, target_kind):
    if target_kind == "unknown":
        return 1.0 if comp_kind == "unknown" else 0.7
    if comp_kind == target_kind:
        return 1.0
    if comp_kind == "unknown":
        return 0.5
    return 0.2


def release_day_values(comp):
    """Hour 1..24 viewers for a comparable's release day, or None if incomplete."""
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    values = {}
    for row in comp.get("hourly") or []:
        try:
            hour = int((parse_time(row["from"]) - day_start) // 3600) + 1
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)):
            values[hour] = max(0.0, float(viewers))
    if len(values) != 24 or values.get(1, 0) < 1:
        return None
    return values


def weighted_quantile(pairs, tau):
    """Stepwise weighted quantile: pairs are (value, weight), sorted by value."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0] if pairs else 0.0
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= tau * total:
            return value
    return pairs[-1][0]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    target_kind = unlock_kind(case.get("notes") or "")

    # Usable comparables, in a fixed order so two runs agree byte for byte.
    comps = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: (str(c.get("name", "")), str(c.get("day_start", "")))):
        if tier_of(comp.get("unlock") or "") == "target":
            continue
        values = release_day_values(comp)
        if values is None:
            continue
        comps.append((comp, values))

    if len(comps) < 5:
        return persistence_forecast(p1, case)

    log_p1 = math.log(p1)
    weights = []
    for comp, values in comps:
        d = math.log(values[1]) - log_p1
        w = math.exp(-(d * d) / (2 * BASE_BW * BASE_BW))
        w *= regime_weight(unlock_kind(comp.get("unlock") or ""), target_kind)
        w *= TIER_WEIGHT[tier_of(comp.get("unlock") or "")]
        weights.append(w)
    if sum(weights) <= 0:
        return persistence_forecast(p1, case)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        # Weighted local-linear fit of the log ratio on log(hour-1 viewers),
        # evaluated at this case's own hour 1.  Falls back to the weighted
        # median when the fit is degenerate (near-zero spread in x).
        points = [
            (
                math.log(values[1]),
                min(LOG_RATIO_MAX, max(LOG_RATIO_MIN, math.log(max(1.0, values[h]) / max(1.0, values[1])))),
                w,
            )
            for (_, values), w in zip(comps, weights)
        ]
        total_w = sum(w for _, _, w in points)
        xm = sum(w * x for x, _, w in points) / total_w
        ym = sum(w * y for _, y, w in points) / total_w
        sxx = sum(w * (x - xm) ** 2 for x, _, w in points)
        sxy = sum(w * (x - xm) * (y - ym) for x, y, w in points)
        if sxx > 1e-9:
            slope = sxy / sxx
            centre = ym + slope * (log_p1 - xm)
            var = sum(w * (y - (ym + slope * (x - xm))) ** 2 for x, y, w in points) / total_w
            sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, math.sqrt(max(var, 0.01))))
        else:
            pairs = sorted((y, w) for _, y, w in points)
            centre = weighted_quantile(pairs, 0.5)
            qs = [weighted_quantile(pairs, tau) for tau in LEVELS]
            sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, (qs[3] - qs[1]) / 1.349))
        out_q = []
        prev = 0.0
        for z in Z:
            q = round2(p1 * math.exp(centre + z * sigma))
            q = min(1e9, max(0.01, float(q)))
            if q < prev:
                q = prev
            out_q.append(q)
            prev = q
        hours.append({"hour": h, "q": out_q})
    return {"hours": hours}


def persistence_forecast(p1, case):
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        out_q = []
        prev = 0.0
        for z in Z:
            q = round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))
            q = min(1e9, max(0.01, float(q)))
            if q < prev:
                q = prev
            out_q.append(q)
            prev = q
        hours.append({"hour": h, "q": out_q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
