#!/usr/bin/env python3
import os, json
import numpy as np

IN_META = "out/cond_cdf_meta.json"
IN_GRID = "out/cond_cdf_grid_10x901x90001.f32"

OUT_DIR = "data"
OUT_META = "data/cond_cdf_meta.json"  # new meta for bucket files

def main():
    if not os.path.exists(IN_META):
        raise SystemExit(f"Missing {IN_META}")
    meta = json.load(open(IN_META))

    if not os.path.exists(IN_GRID):
        raise SystemExit(f"Missing {IN_GRID}")

    B, L, N = meta["shape"]
    B = int(B); L = int(L); N = int(N)
    if B != 10:
        raise SystemExit(f"Expected 10 buckets, got B={B}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # mmap input grid (10, 901, N)
    g = np.memmap(IN_GRID, dtype=np.float32, mode="r", shape=(B, L, N))

    out_files = []
    for b in range(B):
        out_path = os.path.join(OUT_DIR, f"cond_bucket_{b}.f32")
        print("Writing", out_path)

        # create output memmap (901, N)
        out = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(L, N))

        # copy bucket b slice (streamed by OS; not loaded all at once)
        out[:] = g[b, :, :]
        out.flush()
        out_files.append(out_path)

    # write new meta referencing bucket files
    meta2 = dict(meta)
    meta2["grid_file"] = None
    meta2["bucket_files"] = [os.path.basename(p) for p in out_files]  # relative to data/
    meta2["bucket_file_shape"] = [L, N]  # each bucket file is (901, num_bins)

    with open(OUT_META, "w") as f:
        json.dump(meta2, f, indent=2)

    print("Wrote meta:", OUT_META)
    print("Done.")

if __name__ == "__main__":
    main()
