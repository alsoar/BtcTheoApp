#!/usr/bin/env python3
import json
import os

import numpy as np

PRICE_PATH = "out/close_rev.f32"
TWAP_PATH = "out/close_twap60_rev.f32"

OUT_DIR = "data"
OUT_META = os.path.join(OUT_DIR, "cond16_twap60_meta.json")
BUCKET_FILE_FMT = os.path.join(OUT_DIR, "cond16_twap60_bucket_{b}.f32")

MAX_LAG = 900
PAST_LAG = 900
TWAP_POINTS = 60

BP_MIN = -4000.0
BP_MAX = 5000.0
STEP_BP = 0.1

CHUNK = 2_000_000

EDGES = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]


def bucketize_bp_past(bp: np.ndarray) -> np.ndarray:
    b = np.empty(bp.shape, dtype=np.uint8)
    b[:] = 255

    b[bp <= -EDGES[-1]] = 0
    b[(bp > -EDGES[-1]) & (bp <= -EDGES[-2])] = 1
    b[(bp > -EDGES[-2]) & (bp <= -EDGES[-3])] = 2
    b[(bp > -EDGES[-3]) & (bp <= -EDGES[-4])] = 3
    b[(bp > -EDGES[-4]) & (bp <= -EDGES[-5])] = 4
    b[(bp > -EDGES[-5]) & (bp <= -EDGES[-6])] = 5
    b[(bp > -EDGES[-6]) & (bp <= -EDGES[-7])] = 6
    b[(bp > -EDGES[-7]) & (bp <= 0.0)] = 7

    b[(bp > 0.0) & (bp <= EDGES[0])] = 8
    b[(bp > EDGES[0]) & (bp <= EDGES[1])] = 9
    b[(bp > EDGES[1]) & (bp <= EDGES[2])] = 10
    b[(bp > EDGES[2]) & (bp <= EDGES[3])] = 11
    b[(bp > EDGES[3]) & (bp <= EDGES[4])] = 12
    b[(bp > EDGES[4]) & (bp <= EDGES[5])] = 13
    b[(bp > EDGES[5]) & (bp <= EDGES[6])] = 14
    b[bp > EDGES[6]] = 15

    return b


def main():
    if not os.path.exists(PRICE_PATH):
        raise SystemExit(f"Missing {PRICE_PATH}")
    if not os.path.exists(TWAP_PATH):
        raise SystemExit(f"Missing {TWAP_PATH}. Run build_close_twap60_reverse.py first.")

    os.makedirs(OUT_DIR, exist_ok=True)

    x = np.memmap(PRICE_PATH, dtype=np.float32, mode="r")
    twap = np.memmap(TWAP_PATH, dtype=np.float32, mode="r")
    n = x.size
    n_twap = twap.size

    bp10_min = int(round(BP_MIN * 10.0))
    bp10_max = int(round(BP_MAX * 10.0))
    num_bins = bp10_max - bp10_min + 1

    i_min = MAX_LAG
    i_max = n - PAST_LAG - 1
    if i_max <= i_min:
        raise SystemExit("Not enough data.")

    print(f"N prices: {n:,}")
    print(f"N twap values: {n_twap:,}")
    print(f"Valid i: [{i_min}, {i_max}] count={i_max - i_min + 1:,}")

    bucket_path = "out/bucket16_twap60_id_u8.bin"
    bmap = np.memmap(bucket_path, dtype=np.uint8, mode="w+", shape=(n,))
    bmap[:] = 255

    print("Computing bucket ids (past 900s move)...")
    for i0 in range(0, i_max + 1, CHUNK):
        i1 = min(i0 + CHUNK, i_max + 1)

        pt = x[i0:i1].astype(np.float64, copy=False)
        pprev = x[i0 + PAST_LAG:i1 + PAST_LAG].astype(np.float64, copy=False)

        mask = (pprev > 0.0) & np.isfinite(pt) & np.isfinite(pprev)
        bp_past = np.empty(pt.shape, dtype=np.float64)
        bp_past[:] = np.nan
        bp_past[mask] = 10000.0 * (pt[mask] / pprev[mask] - 1.0)

        outb = np.empty(pt.shape, dtype=np.uint8)
        outb[:] = 255
        ok = np.isfinite(bp_past)
        outb[ok] = bucketize_bp_past(bp_past[ok])

        bmap[i0:i1] = outb

        if (i0 // CHUNK) % 10 == 0:
            print(f"  bucket pass: {i1:,}/{i_max + 1:,}")

    bmap.flush()
    print("bucket ids written:", bucket_path)

    b_count = 16
    l_count = MAX_LAG + 1
    bucket_files = []
    outs = []
    for bi in range(b_count):
        path = BUCKET_FILE_FMT.format(b=bi)
        bucket_files.append(os.path.basename(path))
        mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(l_count, num_bins))
        mm[:] = np.nan
        mm.flush()
        outs.append(mm)

    print("Building conditional TWAP CDFs for lags 1..900...")
    for lag in range(1, MAX_LAG + 1):
        start = max(i_min, lag)
        end = min(i_max + 1, n_twap + lag)
        counts_flat = np.zeros(b_count * num_bins, dtype=np.int64)

        for i0 in range(start, end, CHUNK):
            i1 = min(i0 + CHUNK, end)

            cur = x[i0:i1].astype(np.float64, copy=False)
            avg = twap[i0 - lag:i1 - lag].astype(np.float64, copy=False)
            bb = bmap[i0:i1].astype(np.int32, copy=False)

            mask = (bb != 255) & (cur > 0.0) & np.isfinite(cur) & np.isfinite(avg) & (avg > 0.0)
            if not np.any(mask):
                continue

            bp = 10000.0 * (avg[mask] / cur[mask] - 1.0)
            bp10 = np.rint(bp * 10.0).astype(np.int32)
            bp10 = np.clip(bp10, bp10_min, bp10_max)
            col = bp10 - bp10_min
            key = bb[mask] * num_bins + col
            counts_flat += np.bincount(key, minlength=b_count * num_bins)

        counts = counts_flat.reshape(b_count, num_bins)
        for bi in range(b_count):
            total = counts[bi].sum()
            if total == 0:
                outs[bi][lag, :] = np.nan
            else:
                cdf = np.cumsum(counts[bi], dtype=np.float64) / float(total)
                outs[bi][lag, :] = cdf.astype(np.float32)

        if lag % 10 == 0 or lag == 1:
            for mm in outs:
                mm.flush()
            print(f"  finished lag {lag}/{MAX_LAG}")

    for mm in outs:
        mm.flush()

    meta = {
        "meaning": "bucketed-conditional TWAP grid: CDF_b(lag,bp)=P(10000*(mean(P[t+lag-59:t+lag])/P[t]-1)<=bp | bp_past_15m in bucket b)",
        "price_path": PRICE_PATH,
        "twap_path": TWAP_PATH,
        "bucket_id_path": bucket_path,
        "dtype": "float32",
        "bp_min": BP_MIN,
        "bp_max": BP_MAX,
        "step_bp": STEP_BP,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "num_bins": num_bins,
        "twap_points": TWAP_POINTS,
        "bucket_files": bucket_files,
        "bucket_file_shape": [l_count, num_bins],
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
