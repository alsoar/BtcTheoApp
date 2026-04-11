#!/usr/bin/env python3
import json
import os

import numpy as np

PRICE_PATH = "out/close_rev.f32"
OUT_PATH = "out/close_twap60_rev.f32"
OUT_META = "out/close_twap60_meta.json"

TWAP_POINTS = 60
CHUNK_OUTPUTS = 5_000_000


def main():
    if not os.path.exists(PRICE_PATH):
        raise SystemExit(f"Missing {PRICE_PATH}")

    os.makedirs("out", exist_ok=True)

    x = np.memmap(PRICE_PATH, dtype=np.float32, mode="r")
    n = x.size
    if n < TWAP_POINTS:
        raise SystemExit(f"Not enough prices: N={n} < TWAP_POINTS={TWAP_POINTS}")

    out_len = n - TWAP_POINTS + 1
    out = np.memmap(OUT_PATH, dtype=np.float32, mode="w+", shape=(out_len,))

    print(f"N prices: {n:,}")
    print(f"TWAP points: {TWAP_POINTS}")
    print(f"Output values: {out_len:,}")

    for o0 in range(0, out_len, CHUNK_OUTPUTS):
        o1 = min(o0 + CHUNK_OUTPUTS, out_len)
        buf = x[o0:o1 + TWAP_POINTS - 1].astype(np.float64, copy=False)
        cs = np.empty(buf.size + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(buf, out=cs[1:])
        avg = (cs[TWAP_POINTS:] - cs[:-TWAP_POINTS]) / float(TWAP_POINTS)
        out[o0:o1] = avg.astype(np.float32)
        if o0 == 0 or ((o0 // CHUNK_OUTPUTS) + 1) % 10 == 0:
            print(f"  processed {o1:,}/{out_len:,}")

    out.flush()

    meta = {
        "file": OUT_PATH,
        "dtype": "float32",
        "ordering": "reverse chronological, aligned so twap[idx] = mean(close_rev[idx:idx+60])",
        "twap_points": TWAP_POINTS,
        "source_price_path": PRICE_PATH,
        "length": out_len,
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:", OUT_PATH)
    print("Meta:", OUT_META)


if __name__ == "__main__":
    main()
