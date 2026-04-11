#!/usr/bin/env python3
import json
import os

import numpy as np

PRICE_PATH = "out/close_rev.f32"
TWAP_PATH = "out/close_twap60_rev.f32"

OUT_GRID = "data/cdf_grid_twap60.f32"
OUT_META = "data/cdf_grid_twap60_meta.json"

MAX_LAG = 900
TWAP_POINTS = 60

BP_MIN = -4000.0
BP_MAX = 5000.0
STEP_BP = 0.1

CHUNK = 5_000_000


def main():
    if not os.path.exists(PRICE_PATH):
        raise SystemExit(f"Missing {PRICE_PATH}")
    if not os.path.exists(TWAP_PATH):
        raise SystemExit(f"Missing {TWAP_PATH}. Run build_close_twap60_reverse.py first.")

    os.makedirs("data", exist_ok=True)

    x = np.memmap(PRICE_PATH, dtype=np.float32, mode="r")
    twap = np.memmap(TWAP_PATH, dtype=np.float32, mode="r")
    n = x.size
    n_twap = twap.size

    bp10_min = int(round(BP_MIN * 10.0))
    bp10_max = int(round(BP_MAX * 10.0))
    num_bins = bp10_max - bp10_min + 1

    shape = (MAX_LAG + 1, num_bins)
    grid = np.memmap(OUT_GRID, dtype=np.float32, mode="w+", shape=shape)
    grid[0, :] = np.nan

    print(f"N prices: {n:,}")
    print(f"N twap values: {n_twap:,}")

    for lag in range(1, MAX_LAG + 1):
        start = lag
        end = min(n, n_twap + lag)
        counts = np.zeros(num_bins, dtype=np.int64)

        for i0 in range(start, end, CHUNK):
            i1 = min(i0 + CHUNK, end)
            cur = x[i0:i1].astype(np.float64, copy=False)
            avg = twap[i0 - lag:i1 - lag].astype(np.float64, copy=False)

            mask = (cur > 0.0) & np.isfinite(cur) & np.isfinite(avg) & (avg > 0.0)
            if not np.any(mask):
                continue

            bp = 10000.0 * (avg[mask] / cur[mask] - 1.0)
            bp10 = np.rint(bp * 10.0).astype(np.int32)
            bp10 = np.clip(bp10, bp10_min, bp10_max)
            col = bp10 - bp10_min
            counts += np.bincount(col, minlength=num_bins)

        total = counts.sum()
        if total == 0:
            grid[lag, :] = np.nan
        else:
            cdf = np.cumsum(counts, dtype=np.float64) / float(total)
            grid[lag, :] = cdf.astype(np.float32)

        if lag % 10 == 0 or lag == 1:
            grid.flush()
            print(f"  finished lag {lag}/{MAX_LAG}")

    grid.flush()

    meta = {
        "file": OUT_GRID,
        "dtype": "float32",
        "shape": list(shape),
        "bp_min": BP_MIN,
        "bp_max": BP_MAX,
        "step_bp": STEP_BP,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "num_bins": num_bins,
        "lags_used": [1, MAX_LAG],
        "twap_points": TWAP_POINTS,
        "price_path": PRICE_PATH,
        "twap_path": TWAP_PATH,
        "meaning": "cdf_grid_twap60[lag, col] = P(10000*(mean(P[t+lag-59:t+lag])/P[t]-1) <= bp)",
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote grid:", OUT_GRID)
    print("Wrote meta:", OUT_META)


if __name__ == "__main__":
    main()
