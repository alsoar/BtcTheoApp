#!/usr/bin/env python3
import os, json
import numpy as np

# ---------------- CONFIG ----------------
PRICE_PATH = "out/close_rev.f32"  # float32, newest -> oldest
OUT_GRID   = "out/cond_cdf_grid_10x901x90001.f32"
OUT_META   = "out/cond_cdf_meta.json"

MAX_LAG = 900
PAST_LAG = 900  # conditioning window (15 minutes)

# Use same bp range as your unconditional grid (recommended for consistency)
BP_MIN = -4000.0
BP_MAX =  5000.0
STEP_BP = 0.1

# Processing chunk size (indices per chunk). Lower if you see memory pressure.
CHUNK = 2_000_000

# ---------------- Derived constants ----------------
bp10_min = int(round(BP_MIN * 10))
bp10_max = int(round(BP_MAX * 10))
num_bins = bp10_max - bp10_min + 1

# Buckets edges (in bp units)
# We'll implement exactly:
# 0: <= -800
# 1: (-800, -400]
# 2: (-400, -200]
# 3: (-200, -100]
# 4: (-100, 0]
# 5: (0, 100]
# 6: (100, 200]
# 7: (200, 400]
# 8: (400, 800)  (strict upper)
# 9: >= 800
EDGES = [-800.0, -400.0, -200.0, -100.0, 0.0, 100.0, 200.0, 400.0, 800.0]

def bucketize(bp_past: np.ndarray) -> np.ndarray:
    """
    bp_past is float64 array.
    Return uint8 bucket id 0..9, with special handling for >=800 and <800 in bucket 8.
    """
    b = np.empty(bp_past.shape, dtype=np.uint8)

    # bucket 0: <= -800
    b[bp_past <= -800.0] = 0

    # ( -800, -400 ]
    m = (bp_past > -800.0) & (bp_past <= -400.0)
    b[m] = 1

    # ( -400, -200 ]
    m = (bp_past > -400.0) & (bp_past <= -200.0)
    b[m] = 2

    # ( -200, -100 ]
    m = (bp_past > -200.0) & (bp_past <= -100.0)
    b[m] = 3

    # ( -100, 0 ]
    m = (bp_past > -100.0) & (bp_past <= 0.0)
    b[m] = 4

    # ( 0, 100 ]
    m = (bp_past > 0.0) & (bp_past <= 100.0)
    b[m] = 5

    # ( 100, 200 ]
    m = (bp_past > 100.0) & (bp_past <= 200.0)
    b[m] = 6

    # ( 200, 400 ]
    m = (bp_past > 200.0) & (bp_past <= 400.0)
    b[m] = 7

    # ( 400, 800 )  STRICT upper bound
    m = (bp_past > 400.0) & (bp_past < 800.0)
    b[m] = 8

    # >= 800
    b[bp_past >= 800.0] = 9

    return b

def main():
    if not os.path.exists(PRICE_PATH):
        raise SystemExit(f"Missing {PRICE_PATH}. You need out/close_rev.f32 first.")

    os.makedirs("out", exist_ok=True)

    # Price series
    x = np.memmap(PRICE_PATH, dtype=np.float32, mode="r")
    N = x.size
    print(f"N prices: {N:,}")

    # Valid index range for BOTH:
    # past window uses i+900 (older), so i <= N-901
    # forward max lag uses i-900 (newer), so i >= 900
    i_min = MAX_LAG
    i_max = N - PAST_LAG - 1  # inclusive
    if i_max <= i_min:
        raise SystemExit("Not enough data to compute conditional grid with these lags.")

    valid_count = i_max - i_min + 1
    print(f"Valid i range: [{i_min}, {i_max}] ({valid_count:,} points)")

    # ---- Phase 1: compute bucket_id for each i ----
    # We'll store only for i in [0..N-1] but mark invalid as 255.
    bucket_path = "out/bucket_id_u8.bin"
    bucket = np.memmap(bucket_path, dtype=np.uint8, mode="w+", shape=(N,))
    bucket[:] = 255  # default invalid

    print("Computing bucket_id based on past 900s move...")
    for i0 in range(0, i_max + 1, CHUNK):
        i1 = min(i0 + CHUNK, i_max + 1)

        # Need i+900 within bounds => i1+900 <= N
        # Here i1 <= i_max+1 ensures i1+900 <= N
        pt = x[i0:i1].astype(np.float64, copy=False)                # P_t
        pprev = x[i0+PAST_LAG:i1+PAST_LAG].astype(np.float64, copy=False)  # P_{t-900} in real time (older in array)

        # guard
        mask = (pprev != 0.0) & np.isfinite(pt) & np.isfinite(pprev)

        bp_past = np.empty(pt.shape, dtype=np.float64)
        bp_past[:] = np.nan
        bp_past[mask] = 10000.0 * (pt[mask] / pprev[mask] - 1.0)

        b = np.empty(pt.shape, dtype=np.uint8)
        b[:] = 255
        ok = np.isfinite(bp_past)
        b[ok] = bucketize(bp_past[ok])

        bucket[i0:i1] = b

        if (i0 // CHUNK) % 10 == 0:
            print(f"  bucket pass: {i1:,}/{i_max+1:,}")

    bucket.flush()
    print("bucket_id done:", bucket_path)

    # ---- Phase 2: build conditional CDF grid ----
    # Output grid: (10, 901, num_bins), float32
    shape = (10, MAX_LAG + 1, num_bins)
    grid = np.memmap(OUT_GRID, dtype=np.float32, mode="w+", shape=shape)
    grid[:, 0, :] = np.nan  # lag 0 unused
    grid.flush()

    print(f"Building conditional CDF grid -> {OUT_GRID}")
    print(f"Grid shape: {shape} (~{(10*(MAX_LAG+1)*num_bins*4)/1e9:.2f} GB)")

    # For each lag, one pass over valid indices
    # Valid for this lag requires i >= lag and i <= i_max (past window exists already)
    for lag in range(1, MAX_LAG + 1):
        start = max(i_min, lag)
        end = i_max  # inclusive

        counts_flat = np.zeros(10 * num_bins, dtype=np.int64)

        # process in chunks over i
        for i0 in range(start, end + 1, CHUNK):
            i1 = min(i0 + CHUNK, end + 1)

            # forward return over lag seconds:
            # In reverse array: newer is i-lag
            older = x[i0:i1].astype(np.float64, copy=False)         # P_t (older in real time)
            newer = x[i0-lag:i1-lag].astype(np.float64, copy=False) # P_{t+lag} (newer in real time)

            b = bucket[i0:i1].astype(np.int32, copy=False)          # 0..9 or 255

            mask = (b != 255) & (older != 0.0) & np.isfinite(older) & np.isfinite(newer)
            if not np.any(mask):
                continue

            bp_fwd = 10000.0 * (newer[mask] / older[mask] - 1.0)
            bp10 = np.rint(bp_fwd * 10.0).astype(np.int32)

            # clamp to grid range
            bp10 = np.clip(bp10, bp10_min, bp10_max)
            col = bp10 - bp10_min  # 0..num_bins-1

            bb = b[mask]  # 0..9
            key = bb * num_bins + col  # 0..10*num_bins-1

            # count
            counts_flat += np.bincount(key, minlength=10 * num_bins)

        # reshape into (10, num_bins)
        counts = counts_flat.reshape(10, num_bins)

        # convert to CDF per bucket, store as float32
        for bi in range(10):
            total = counts[bi].sum()
            if total == 0:
                # fallback: if no samples, set to unconditional-like flat NaNs
                grid[bi, lag, :] = np.nan
                continue
            cdf = np.cumsum(counts[bi], dtype=np.float64) / float(total)
            grid[bi, lag, :] = cdf.astype(np.float32)

        grid.flush()
        if lag % 10 == 0 or lag == 1:
            print(f"  finished lag {lag}/{MAX_LAG}")

    meta = {
        "price_path": PRICE_PATH,
        "bucket_id_path": bucket_path,
        "meaning": "grid[bucket, lag, bp_bin] = P(bp_future <= bp_value | bp_past_15m in bucket)",
        "grid_file": OUT_GRID,
        "dtype": "float32",
        "shape": list(shape),
        "bp_min": BP_MIN,
        "bp_max": BP_MAX,
        "step_bp": STEP_BP,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "num_bins": num_bins,
        "past_lag_seconds": PAST_LAG,
        "future_lags_seconds": [1, MAX_LAG],
        "buckets": [
            {"id": 0, "label": "bp_past <= -800"},
            {"id": 1, "label": "-800 < bp_past <= -400"},
            {"id": 2, "label": "-400 < bp_past <= -200"},
            {"id": 3, "label": "-200 < bp_past <= -100"},
            {"id": 4, "label": "-100 < bp_past <= 0"},
            {"id": 5, "label": "0 < bp_past <= 100"},
            {"id": 6, "label": "100 < bp_past <= 200"},
            {"id": 7, "label": "200 < bp_past <= 400"},
            {"id": 8, "label": "400 < bp_past < 800"},
            {"id": 9, "label": "bp_past >= 800"},
        ],
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("DONE")
    print("Grid:", OUT_GRID)
    print("Meta:", OUT_META)
    print("Bucket IDs:", bucket_path)

if __name__ == "__main__":
    main()
