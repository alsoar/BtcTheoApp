import csv, json, os, math
import numpy as np

INP = "out/bp_lags_1_900_cdf.csv"
OUT_GRID = "out/cdf_grid.f32"
OUT_META = "out/cdf_grid_meta.json"

# Grid spec (0.1 bp resolution)
BP_MIN = -4000.0
BP_MAX =  5000.0
STEP_BP = 0.1

bp10_min = int(round(BP_MIN * 10))
bp10_max = int(round(BP_MAX * 10))
num_bins = bp10_max - bp10_min + 1

# Rows: 0..900, but we use 1..900 (row 0 left as NaN)
max_lag = 900
shape = (max_lag + 1, num_bins)

os.makedirs("out", exist_ok=True)

# Create a memmap on disk (RAM-safe)
grid = np.memmap(OUT_GRID, dtype=np.float32, mode="w+", shape=shape)

# Initialize row 0 as NaN (unused)
grid[0, :] = np.nan

def fill_row_from_sparse(lag: int, sparse_bp10: list[int], sparse_cdf: list[float]) -> None:
    """
    Builds a fully-filled CDF row for one lag across all bp10 bins.
    - For bins below the first observed bp: CDF = 0
    - Between observed bp bins: carry forward last CDF (step function)
    - At observed bp bin: set CDF to that row's cdf
    - Above last observed bp: CDF = 1
    """
    row = grid[lag, :]
    row[:] = 0.0  # default below first observed

    cur = 0.0
    last_bin_filled = -1  # relative bin index (0..num_bins-1)

    for bp10, c in zip(sparse_bp10, sparse_cdf):
        if bp10 < bp10_min or bp10 > bp10_max:
            continue

        b = bp10 - bp10_min  # 0..num_bins-1

        # Fill gaps (missing bins) with current CDF
        if b > last_bin_filled + 1:
            row[last_bin_filled+1:b] = cur

        # Clamp and enforce monotone
        c = float(c)
        if not math.isfinite(c):
            continue
        c = max(0.0, min(1.0, c))
        if c < cur:
            c = cur

        # Set current bin
        row[b] = c
        cur = c
        last_bin_filled = b

    # Fill remaining bins above last observed with 1.0
    if last_bin_filled < num_bins - 1:
        row[last_bin_filled+1:] = 1.0

# Stream through CSV once; build each lag row
current_lag = None
s_bp10 = []
s_cdf = []

def flush():
    global current_lag, s_bp10, s_cdf
    if current_lag is None:
        return
    fill_row_from_sparse(current_lag, s_bp10, s_cdf)
    # ensure written to disk progressively
    grid.flush()
    print(f"filled lag {current_lag}")
    s_bp10 = []
    s_cdf = []

with open(INP, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        lag = int(row["lag_seconds"])
        if lag < 1 or lag > max_lag:
            continue

        if current_lag is None:
            current_lag = lag

        if lag != current_lag:
            flush()
            current_lag = lag

        bp = float(row["bp_move"])
        c  = float(row["cdf"])
        s_bp10.append(int(round(bp * 10)))
        s_cdf.append(c)

flush()

meta = {
    "file": OUT_GRID,
    "dtype": "float32",
    "shape": list(shape),
    "lags_used": [1, 900],
    "bp_min": BP_MIN,
    "bp_max": BP_MAX,
    "step_bp": STEP_BP,
    "bp10_min": bp10_min,
    "bp10_max": bp10_max,
    "num_bins": num_bins,
    "meaning": "cdf_grid[lag, bp10-bp10_min] = P(900? no: lag-second return in bp <= bp_move)"
}
with open(OUT_META, "w") as f:
    json.dump(meta, f, indent=2)

print("Wrote grid:", OUT_GRID)
print("Wrote meta:", OUT_META)
