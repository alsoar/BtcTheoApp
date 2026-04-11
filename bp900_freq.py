import os
import numpy as np
from collections import defaultdict
import csv

PATH = "out/close_rev.f32"          # float32 binary, newest -> oldest
OUT  = "out/bp900_freq.csv"         # output frequency table
LAG  = 900                          # seconds
CHUNK = 10_000_000                  # number of points per chunk (tune if you want)

# Open as memmap (doesn't load whole file into RAM)
x = np.memmap(PATH, dtype=np.float32, mode="r")
N = x.size
if N <= LAG:
    raise SystemExit(f"Not enough data: N={N} <= LAG={LAG}")

counts = defaultdict(int)

# i ranges over older indices [LAG, N)
start = LAG
end = N

print(f"N={N:,} values. Computing bp moves with lag={LAG} over {N-LAG:,} points...")
print(f"Chunk size: {CHUNK:,}")

for i0 in range(start, end, CHUNK):
    i1 = min(i0 + CHUNK, end)

    older = x[i0:i1].astype(np.float64, copy=False)            # P_t (older)
    newer = x[i0 - LAG:i1 - LAG].astype(np.float64, copy=False) # P_{t+900} (newer)

    # Guard against divide-by-zero and non-finite values
    mask = (older != 0.0) & np.isfinite(older) & np.isfinite(newer)
    if not np.any(mask):
        continue

    # bp = 10000 * (newer/older - 1)
    bp = 10000.0 * (newer[mask] / older[mask] - 1.0)

    # round to 1 decimal bp by converting to "tenths of bp" integers
    bp10 = np.rint(bp * 10.0).astype(np.int32)

    # Update counts efficiently
    vals, cts = np.unique(bp10, return_counts=True)
    for v, c in zip(vals, cts):
        counts[int(v)] += int(c)

    if (i0 - start) // CHUNK % 5 == 0:
        done = i1 - start
        total = end - start
        print(f"  processed {done:,}/{total:,} points...")

# Write CSV sorted by bp value
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bp_move", "count"])  # bp_move is rounded to 1 decimal
    for bp10 in sorted(counts.keys()):
        w.writerow([bp10 / 10.0, counts[bp10]])

print("Wrote:", OUT)
print("Unique bp bins:", len(counts))
