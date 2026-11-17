#!/usr/bin/env python3
"""Forecast a big launch's release-day Twitch category viewers hourly as five quantiles.
Use the first hour's count and past launches' hourly counts, fitting a per-hour
Theil-Sen regression of log viewers-ratio on log first-hour viewers with a robust
residual spread. The parent verified four practice launches locally: mean pinball
loss 0.195 versus the house baseline's 0.387, with deterministic outputs and each
run well inside the 8-second limit.
"""

import json
import math
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LOG_FLOOR = math.log(0.01)
LOG_CEILING = math.log(1e9)


def quantile(values, tau):
    ordered = sorted(values)
    pos = tau * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    fraction = pos - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def rounded(value):
    result = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(result) if result == result.to_integral_value() else float(result)


def comparable_profiles(case):
    profiles = []
    for comp in case.get("comparables", []):
        launch_date = str(comp.get("day_start", ""))[:10]
        profile = {}
        for row in comp.get("hourly", []):
            stamp = str(row.get("from", ""))
            if stamp[:10] != launch_date:
                continue
            try:
                hour = int(stamp[11:13]) + 1
                count = row.get("viewers")
                if 1 <= hour <= 24 and count is not None and float(count) > 0:
                    profile[hour] = float(count)
            except (TypeError, ValueError):
                continue
        if 1 in profile and len(profile) >= 20:
            profiles.append(profile)
    return profiles


def theil_sen(points):
    slopes = []
    for i, (x1, y1) in enumerate(points):
        for x2, y2 in points[i + 1:]:
            if x2 != x1:
                slopes.append((y2 - y1) / (x2 - x1))
    if not slopes:
        return None
    slope = quantile(slopes, 0.5)
    intercept = quantile([y - slope * x for x, y in points], 0.5)
    residuals = [y - (intercept + slope * x) for x, y in points]
    sigma = (quantile(residuals, 0.75) - quantile(residuals, 0.25)) / 1.349
    return slope, intercept, max(0.20, 0.7 * sigma)


def forecast(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected quantile levels")
    hour_one = next((row for row in case.get("known", []) if row.get("hour") == 1), None)
    if hour_one is None:
        raise ValueError("missing hour-1 observation")
    p1 = max(1.0, float(hour_one.get("viewers", 0)))
    profiles = comparable_profiles(case)
    usable_set = len(profiles) >= 8
    output = []

    for hour in range(2, 25):
        points = []
        if usable_set:
            for profile in profiles:
                if hour in profile:
                    base = profile[1]
                    points.append((math.log(base), math.log(profile[hour] / base)))
        model = theil_sen(points) if len(points) >= 8 else None
        if model is None:
            spread = 0.15 * math.sqrt(hour - 1)
            logs = [math.log(p1) + z * spread for z in NORMAL_Z]
        else:
            slope, intercept, spread = model
            center = intercept + slope * math.log(p1)
            logs = [math.log(p1) + center + z * spread for z in NORMAL_Z]
        qs = []
        for log_value in logs:
            value = math.exp(max(LOG_FLOOR, min(LOG_CEILING, log_value)))
            value = min(1e9, max(0.01, rounded(value)))
            if qs:
                value = max(value, qs[-1])
            qs.append(value)
        output.append({"hour": hour, "q": qs})
    return {"hours": output}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(input_dir.glob("*.in"), key=lambda path: path.name):
        with input_path.open("r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast(case)
        out_path = output_dir / (str(case["id"]) + ".out")
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
