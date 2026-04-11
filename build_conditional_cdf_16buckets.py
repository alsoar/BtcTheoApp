#!/usr/bin/env python3
import os, json
import numpy as np

# ---------------- INPUTS ----------------
PRICE_PATH = "out/close_rev.f32"   # float32, newest -> oldest

# ---------------- OUTPUTS ----------------
OUT_DIR  = "data"
OUT_META = os.path.join(OUT_DIR, "cond16_cdf_meta.json")
BUCKET_FILE_FMT = os.path.join(OUT_DIR, "cond16_bucket_{b}.f32")

# ---------------- SETTINGS ----------------
MAX_LAG = 900
PAST_LAG = 900  # conditioning window (15 minutes)

# bp grid (keep consistent with your existing grids)
BP_MIN = -4000.0
BP_MAX =  5000.0
STEP_BP = 0.1

# chunk size (tune if needed)
CHUNK = 2_000_000

bp10_min = int(round(BP_MIN * 10))
bp10_max = int(round(BP_MAX * 10))
num_bins = bp10_max - bp10_min + 1

# Bucket edges (magnitudes)
EDGES = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]

def bucketize_bp_past(bp: np.ndarray) -> np.ndarray:
    """
    bp is float64 array of bp_past values.
    Returns uint8 bucket id 0..15 following the spec.
    """
    b = np.empty(bp.shape, dtype=np.uint8)
    b[:] = 255

    # Negative / down side (including 0)
    b[bp <= -EDGES[-1]] = 0                              # <= -800
    b[(bp > -EDGES[-1]) & (bp <= -EDGES[-2])] = 1        # (-800, -400]
    b[(bp > -EDGES[-2]) & (bp <= -EDGES[-3])] = 2        # (-400, -200]
    b[(bp > -EDGES[-3]) & (bp <= -EDGES[-4])] = 3        # (-200, -100]
    b[(bp > -EDGES[-4]) & (bp <= -EDGES[-5])] = 4        # (-100, -50]
    b[(bp > -EDGES[-5]) & (bp <= -EDGES[-6])] = 5        # (-50, -25]
    b[(bp > -EDGES[-6]) & (bp <= -EDGES[-7])] = 6        # (-25, -10]
    b[(bp > -EDGES[-7]) & (bp <= 0.0)] = 7               # (-10, 0]

    # Positive / up side
    b[(bp > 0.0) & (bp <= EDGES[0])] = 8                 # (0, 10]
    b[(bp > EDGES[0]) & (bp <= EDGES[1])] = 9            # (10, 25]
    b[(bp > EDGES[1]) & (bp <= EDGES[2])] = 10           # (25, 50]
    b[(bp > EDGES[2]) & (bp <= EDGES[3])] = 11           # (50, 100]
    b[(bp > EDGES[3]) & (bp <= EDGES[4])] = 12           # (100, 200]
    b[(bp > EDGES[4]) & (bp <= EDGES[5])] = 13           # (200, 400]
    b[(bp > EDGES[5]) & (bp <= EDGES[6])] = 14           # (400, 800]
    b[bp > EDGES[6]] = 15                                # > 800

    return b

def main():
    if not os.path.exists(PRICE_PATH):
        raise SystemExit(f"Missing {PRICE_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)

    x = np.memmap(PRICE_PATH, dtype=np.float32, mode="r")
    N = x.size
    print(f"N prices: {N:,}")

    # Valid index range:
    # past needs i+900 in bounds => i <= N-901
    # future needs i-lag in bounds => i >= lag (max lag 900 => i>=900)
    i_min = MAX_LAG
    i_max = N - PAST_LAG - 1
    if i_max <= i_min:
        raise SystemExit("Not enough data.")
    print(f"Valid i: [{i_min}, {i_max}]  count={i_max - i_min + 1:,}")

    # ---- bucket_id array (u8) stored in out/ for debugging; not required by app ----
    bucket_path = "out/bucket16_id_u8.bin"
    bmap = np.memmap(bucket_path, dtype=np.uint8, mode="w+", shape=(N,))
    bmap[:] = 255

    print("Computing bucket ids (past 900s move)...")
    for i0 in range(0, i_max + 1, CHUNK):
        i1 = min(i0 + CHUNK, i_max + 1)

        pt = x[i0:i1].astype(np.float64, copy=False)
        pprev = x[i0+PAST_LAG:i1+PAST_LAG].astype(np.float64, copy=False)

        mask = (pprev != 0.0) & np.isfinite(pt) & np.isfinite(pprev)
        bp_past = np.empty(pt.shape, dtype=np.float64)
        bp_past[:] = np.nan
        bp_past[mask] = 10000.0 * (pt[mask] / pprev[mask] - 1.0)

        outb = np.empty(pt.shape, dtype=np.uint8)
        outb[:] = 255
        ok = np.isfinite(bp_past)
        outb[ok] = bucketize_bp_past(bp_past[ok])

        bmap[i0:i1] = outb

        if (i0 // CHUNK) % 10 == 0:
            print(f"  bucket pass: {i1:,}/{i_max+1:,}")

    bmap.flush()
    print("bucket ids written:", bucket_path)

    # ---- Create 16 output bucket files (each shape 901 x num_bins) ----
    B = 16
    L = MAX_LAG + 1
    bucket_files = []
    outs = []
    for bi in range(B):
        p = BUCKET_FILE_FMT.format(b=bi)
        bucket_files.append(os.path.basename(p))
        mm = np.memmap(p, dtype=np.float32, mode="w+", shape=(L, num_bins))
        mm[:] = np.nan
        mm.flush()
        outs.append(mm)
    print("Created 16 bucket files in data/")

    print("Building conditional CDFs for lags 1..900...")
    for lag in range(1, MAX_LAG + 1):
        start = max(i_min, lag)
        end = i_max

        counts_flat = np.zeros(B * num_bins, dtype=np.int64)

        for i0 in range(start, end + 1, CHUNK):
            i1 = min(i0 + CHUNK, end + 1)

            older = x[i0:i1].astype(np.float64, copy=False)          # P_t
            newer = x[i0-lag:i1-lag].astype(np.float64, copy=False)  # P_{t+lag}
            bb = bmap[i0:i1].astype(np.int32, copy=False)

            mask = (bb != 255) & (older != 0.0) & np.isfinite(older) & np.isfinite(newer)
            if not np.any(mask):
                continue

            bp_fwd = 10000.0 * (newer[mask] / older[mask] - 1.0)
            bp10 = np.rint(bp_fwd * 10.0).astype(np.int32)
            bp10 = np.clip(bp10, bp10_min, bp10_max)
            col = bp10 - bp10_min

            key = bb[mask] * num_bins + col
            counts_flat += np.bincount(key, minlength=B * num_bins)

        counts = counts_flat.reshape(B, num_bins)

        # write CDF row for each bucket at this lag
        for bi in range(B):
            tot = counts[bi].sum()
            if tot == 0:
                outs[bi][lag, :] = np.nan
            else:
                cdf = np.cumsum(counts[bi], dtype=np.float64) / float(tot)
                outs[bi][lag, :] = cdf.astype(np.float32)

        # flush occasionally
        if lag % 10 == 0 or lag == 1:
            for bi in range(B):
                outs[bi].flush()
            print(f"  finished lag {lag}/{MAX_LAG}")

    # final flush
    for bi in range(B):
        outs[bi].flush()

    meta = {
        "meaning": "bucketed-conditional grid: CDF_b(lag,bp)=P(bp_future<=bp | bp_past_15m in bucket b)",
        "price_path": PRICE_PATH,
        "bucket_id_path": bucket_path,
        "dtype": "float32",
        "bp_min": BP_MIN,
        "bp_max": BP_MAX,
        "step_bp": STEP_BP,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "num_bins": num_bins,
        "bucket_files": bucket_files,
        "bucket_file_shape": [L, num_bins],
        "past_lag_seconds": PAST_LAG,
        "future_lags_seconds": [1, MAX_LAG],
        "edges_bp": EDGES,
        "bucket_scheme": [
            "0: bp<=-800",
            "1: -800<bp<=-400",
            "2: -400<bp<=-200",
            "3: -200<bp<=-100",
            "4: -100<bp<=-50",
            "5: -50<bp<=-25",
            "6: -25<bp<=-10",
            "7: -10<bp<=0",
            "8: 0<bp<=10",
            "9: 10<bp<=25",
            "10: 25<bp<=50",
            "11: 50<bp<=100",
            "12: 100<bp<=200",
            "13: 200<bp<=400",
            "14: 400<bp<=800",
            "15: bp>800",
        ],
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote meta:", OUT_META)
    print("DONE")

if __name__ == "__main__":
    main()
