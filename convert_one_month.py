import subprocess
import numpy as np
import os

ZIP = "zips/BTCUSDT-1s-2026-01.zip"
OUT_RAW = "out/BTCUSDT_close_1s_2026-01_rev.f32"
OUT_ZST = OUT_RAW + ".zst"

os.makedirs("out", exist_ok=True)

cmd = f'unzip -p "{ZIP}" | awk -F"," \'{{print $5}}\''
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)

data = np.fromstring(p.stdout.read(), sep="\n", dtype=np.float32)
rc = p.wait()
if rc != 0:
    raise RuntimeError("unzip/awk failed")

# fully reverse: most recent second first
data = data[::-1]

with open(OUT_RAW, "wb") as f:
    f.write(data.tobytes(order="C"))

subprocess.check_call(["zstd", "-19", "-f", "--rm", OUT_RAW, "-o", OUT_ZST])

print("Wrote:", OUT_ZST)
print("Values:", data.size)
