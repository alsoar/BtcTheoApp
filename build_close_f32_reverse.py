import os, glob, json, subprocess
import numpy as np

ZIPS_DIR = "zips"
OUT_RAW = "out/close_rev.f32"
OUT_ZST = "out/close_rev.f32.zst"
OFFSETS = "out/month_offsets.csv"
META = "out/meta.json"

zips = sorted(glob.glob(os.path.join(ZIPS_DIR, "BTCUSDT-1s-*.zip")), reverse=True)
if not zips:
    raise SystemExit("No zips found in ./zips (expected BTCUSDT-1s-YYYY-MM.zip files)")

os.makedirs("out", exist_ok=True)

offset = 0
with open(OUT_RAW, "wb") as out, open(OFFSETS, "w") as idx:
    idx.write("month,byte_offset,n_values\n")

    for z in zips:
        base = os.path.basename(z)             # BTCUSDT-1s-YYYY-MM.zip
        ym = base.split("-")[-1].replace(".zip","")

        # Stream unzip -> extract Close (column 5)
        cmd = f'unzip -p "{z}" | awk -F"," \'{{print $5}}\''
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)

        # Parse newline-separated floats efficiently
        data = np.fromstring(p.stdout.read(), sep="\n", dtype=np.float32)

        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Failed processing {z} (rc={rc})")

        # FULLY reverse within the month so overall file is newest-second-first
        data = data[::-1]

        n = data.size
        out.write(data.tobytes(order="C"))

        idx.write(f"{ym},{offset},{n}\n")
        offset += n * 4  # float32 = 4 bytes

# Compress the big binary output
subprocess.check_call(["zstd", "-19", "-f", "--rm", OUT_RAW, "-o", OUT_ZST])

meta = {
    "symbol": "BTCUSDT",
    "interval": "1s",
    "field": "close (column 5)",
    "dtype": "float32",
    "ordering": "fully reverse chronological (most recent second first)",
    "compressed_file": OUT_ZST,
    "index_file": OFFSETS
}
with open(META, "w") as f:
    json.dump(meta, f, indent=2)

print("DONE. Wrote:", OUT_ZST)
print("Index:", OFFSETS)
print("Meta:", META)
