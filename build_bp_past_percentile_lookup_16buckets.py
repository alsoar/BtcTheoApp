import os, json
import numpy as np

PRICE_PATH = "out/close_rev.f32"
OUT_TABLE = "data/bp_past_pct_16x90001.f16"
OUT_META  = "data/bp_past_pct_meta.json"

PAST_LAG = 900
CHUNK = 2_000_000

BP_MIN = -4000.0
BP_MAX =  5000.0

bp10_min = int(round(BP_MIN * 10))
bp10_max = int(round(BP_MAX * 10))
nbins = bp10_max - bp10_min + 1

edges = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]

def bucketize_bp_past(bp):
    b = np.empty(bp.shape, dtype=np.uint8)
    b[:] = 255

    b[bp <= -edges[-1]] = 0
    b[(bp > -edges[-1]) & (bp <= -edges[-2])] = 1
    b[(bp > -edges[-2]) & (bp <= -edges[-3])] = 2
    b[(bp > -edges[-3]) & (bp <= -edges[-4])] = 3
    b[(bp > -edges[-4]) & (bp <= -edges[-5])] = 4
    b[(bp > -edges[-5]) & (bp <= -edges[-6])] = 5
    b[(bp > -edges[-6]) & (bp <= -edges[-7])] = 6
    b[(bp > -edges[-7]) & (bp <= 0.0)] = 7

    b[(bp > 0.0) & (bp <= edges[0])] = 8
    b[(bp > edges[0]) & (bp <= edges[1])] = 9
    b[(bp > edges[1]) & (bp <= edges[2])] = 10
    b[(bp > edges[2]) & (bp <= edges[3])] = 11
    b[(bp > edges[3]) & (bp <= edges[4])] = 12
    b[(bp > edges[4]) & (bp <= edges[5])] = 13
    b[(bp > edges[5]) & (bp <= edges[6])] = 14
    b[bp > edges[6]] = 15

    return b

def main():
    if not os.path.exists(PRICE_PATH):
        raise SystemExit(f"Missing {PRICE_PATH}")

    os.makedirs("data", exist_ok=True)

    x = np.memmap(PRICE_PATH, dtype=np.float32, mode="r")
    N = x.size
    i_max = N - PAST_LAG - 1
    if i_max < 0:
        raise SystemExit("Not enough data.")

    counts_flat = np.zeros(16 * nbins, dtype=np.int64)

    for i0 in range(0, i_max + 1, CHUNK):
        i1 = min(i0 + CHUNK, i_max + 1)

        pt = x[i0:i1].astype(np.float64, copy=False)
        pprev = x[i0+PAST_LAG:i1+PAST_LAG].astype(np.float64, copy=False)

        mask = (pprev != 0.0) & np.isfinite(pt) & np.isfinite(pprev)
        if not np.any(mask):
            continue

        bp = 10000.0 * (pt[mask] / pprev[mask] - 1.0)
        b = bucketize_bp_past(bp).astype(np.int32)

        bp10 = np.rint(bp * 10.0).astype(np.int32)
        bp10 = np.clip(bp10, bp10_min, bp10_max)
        col = bp10 - bp10_min

        key = b * nbins + col
        counts_flat += np.bincount(key, minlength=16 * nbins)

    counts = counts_flat.reshape(16, nbins)
    totals = counts.sum(axis=1).astype(np.float64)

    pct = np.memmap(OUT_TABLE, dtype=np.float16, mode="w+", shape=(16, nbins))
    pct[:] = np.nan

    for bi in range(16):
        tot = totals[bi]
        if tot <= 0:
            continue
        cdf = np.cumsum(counts[bi], dtype=np.float64) / tot
        pct[bi, :] = cdf.astype(np.float16)

    pct.flush()

    meta = {
        "meaning": "pct[bucket, bp10-bp10_min] = P(bp_past <= bp | bp_past in bucket)",
        "price_path": PRICE_PATH,
        "past_lag_seconds": PAST_LAG,
        "bp_min": BP_MIN,
        "bp_max": BP_MAX,
        "step_bp": 0.1,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "nbins": nbins,
        "dtype": "float16",
        "shape": [16, nbins],
        "file": OUT_TABLE,
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
            "15: bp>800"
        ]
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", OUT_TABLE)
    print("Wrote:", OUT_META)

if __name__ == "__main__":
    main()
