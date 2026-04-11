import os
import csv
import numpy as np
from collections import defaultdict

PATH = "out/close_rev.f32"                 # float32 binary, newest -> oldest
OUT  = "out/bp_lags_1_900_counts.csv"      # master output
MAX_LAG = 900
CHUNK = 5_000_000                          # points per chunk (safe on 16GB; raise to 10M if you want)

x = np.memmap(PATH, dtype=np.float32, mode="r")
N = x.size

if N <= MAX_LAG:
    raise SystemExit(f"Not enough data: N={N} <= MAX_LAG={MAX_LAG}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["lag_seconds", "bp_move", "count"])  # bp_move rounded to 0.1 bp

    for lag in range(1, MAX_LAG + 1):
        counts = defaultdict(int)

        start = lag
        end = N

        # process in chunks
        for i0 in range(start, end, CHUNK):
            i1 = min(i0 + CHUNK, end)

            older = x[i0:i1].astype(np.float64, copy=False)                 # P_t
            newer = x[i0 - lag:i1 - lag].astype(np.float64, copy=False)     # P_{t+lag}

            mask = (older != 0.0) & np.isfinite(older) & np.isfinite(newer)
            if not np.any(mask):
                continue

            bp = 10000.0 * (newer[mask] / older[mask] - 1.0)
            bp10 = np.rint(bp * 10.0).astype(np.int32)  # tenths of bp as int

            vals, cts = np.unique(bp10, return_counts=True)
            for v, c in zip(vals, cts):
                counts[int(v)] += int(c)

        # write this lag to master file, sorted by bp
        for bp10 in sorted(counts.keys()):
            w.writerow([lag, bp10 / 10.0, counts[bp10]])

        print(f"Finished lag {lag}/{MAX_LAG}. Unique bins: {len(counts)}")

print("Wrote:", OUT)
