import json
import math
import mmap
import os
import struct
import streamlit as st

UN_GRID_PATH  = "data/cdf_grid.f32"
UN_META_PATH  = "data/cdf_grid_meta.json"

COND_META_PATH = "data/cond16_cdf_meta.json"
COND_DIR = "data"

PCT_META_PATH = "data/bp_past_pct_meta.json"

STRICT_EPS_BP = 0.1

@st.cache_resource
def load_uncond():
    m = json.load(open(UN_META_PATH))
    rows, cols = map(int, m["shape"])
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    f = open(UN_GRID_PATH, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"rows": rows, "cols": cols, "bp10_min": bp10_min, "bp10_max": bp10_max, "f": f, "mm": mm}

@st.cache_resource
def load_cond_meta():
    m = json.load(open(COND_META_PATH))
    L, N = map(int, m["bucket_file_shape"])
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    bucket_files = m["bucket_files"]
    return {"L": L, "N": N, "bp10_min": bp10_min, "bp10_max": bp10_max, "bucket_files": bucket_files}

@st.cache_resource
def load_bucket_file(bucket: int):
    cm = load_cond_meta()
    B = len(cm["bucket_files"])
    b = max(0, min(bucket, B - 1))
    fname = cm["bucket_files"][b]
    path = os.path.join(COND_DIR, fname)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"path": path, "f": f, "mm": mm}

@st.cache_resource
def load_pct():
    m = json.load(open(PCT_META_PATH))
    file_path = m["file"]
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    nbins = int(m["nbins"])
    f = open(file_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"bp10_min": bp10_min, "bp10_max": bp10_max, "nbins": nbins, "f": f, "mm": mm}

def cdf_uncond(g, lag: int, bp: float) -> float:
    rows, cols = g["rows"], g["cols"]
    bp10_min, bp10_max = g["bp10_min"], g["bp10_max"]
    mm = g["mm"]
    lag = max(1, min(lag, rows - 1))
    bp10 = int(round(bp * 10.0))
    if bp10 < bp10_min: return 0.0
    if bp10 > bp10_max: return 1.0
    col = bp10 - bp10_min
    idx = lag * cols + col
    return struct.unpack_from("<f", mm, idx * 4)[0]

def cdf_cond(bucket: int, lag: int, bp: float) -> float:
    cm = load_cond_meta()
    L, N = cm["L"], cm["N"]
    bp10_min, bp10_max = cm["bp10_min"], cm["bp10_max"]
    B = len(cm["bucket_files"])
    b = max(0, min(bucket, B - 1))
    lag = max(1, min(lag, L - 1))
    bp10 = int(round(bp * 10.0))
    if bp10 < bp10_min: return 0.0
    if bp10 > bp10_max: return 1.0
    col = bp10 - bp10_min
    bf = load_bucket_file(b)
    idx = lag * N + col
    return struct.unpack_from("<f", bf["mm"], idx * 4)[0]

def bucket_id_16(bp: float) -> int:
    if bp <= -800: return 0
    if bp <= -400: return 1
    if bp <= -200: return 2
    if bp <= -100: return 3
    if bp <= -50:  return 4
    if bp <= -25:  return 5
    if bp <= -10:  return 6
    if bp <= 0:    return 7
    if bp <= 10:   return 8
    if bp <= 25:   return 9
    if bp <= 50:   return 10
    if bp <= 100:  return 11
    if bp <= 200:  return 12
    if bp <= 400:  return 13
    if bp <= 800:  return 14
    return 15

BUCKET_LABELS_16 = [
    "≤ -800","(-800, -400]","(-400, -200]","(-200, -100]","(-100, -50]","(-50, -25]","(-25, -10]","(-10, 0]",
    "(0, 10]","(10, 25]","(25, 50]","(50, 100]","(100, 200]","(200, 400]","(400, 800]","> 800"
]

def percentile_in_bucket(bp_past: float, bucket: int) -> float:
    p = load_pct()
    bp10_min, bp10_max, nbins = p["bp10_min"], p["bp10_max"], p["nbins"]
    bp10 = int(round(bp_past * 10.0))
    if bp10 < bp10_min: bp10 = bp10_min
    if bp10 > bp10_max: bp10 = bp10_max
    col = bp10 - bp10_min
    b = max(0, min(bucket, 15))
    off = (b * nbins + col) * 2
    u = struct.unpack_from("<e", p["mm"], off)[0]
    if not math.isfinite(float(u)):
        return 0.5
    u = float(u)
    if u < 0.0: u = 0.0
    if u > 1.0: u = 1.0
    return u

def weights_softmax_3(u: float, tau: float, tau_c: float):
    dL = u
    dU = 1.0 - u
    dC = abs(u - 0.5)
    sL = -dL / max(1e-9, tau)
    sU = -dU / max(1e-9, tau)
    sC = -dC / max(1e-9, tau_c)
    m = max(sL, sC, sU)
    eL = math.exp(sL - m)
    eC = math.exp(sC - m)
    eU = math.exp(sU - m)
    Z = eL + eC + eU
    return eL / Z, eC / Z, eU / Z

def theo_up_from_bps_up_now(cdf_func, lag: int, bp_up_now: float) -> float:
    r_now = bp_up_now / 10000.0
    if (not math.isfinite(r_now)) or (1.0 + r_now) <= 0.0:
        return float("nan")
    bp_req = 10000.0 * ((1.0 / (1.0 + r_now)) - 1.0)
    c = cdf_func(lag, bp_req - STRICT_EPS_BP)
    t = 1.0 - c
    return min(1.0, max(0.0, t))

def theo_up_from_prices(cdf_func, lag: int, s_cur: float, s_target: float) -> float:
    if s_cur <= 0.0 or s_target <= 0.0 or (not math.isfinite(s_cur)) or (not math.isfinite(s_target)):
        return float("nan")
    bp_req = 10000.0 * (s_target / s_cur - 1.0)
    c = cdf_func(lag, bp_req - STRICT_EPS_BP)
    t = 1.0 - c
    return min(1.0, max(0.0, t))

st.set_page_config(page_title="YES Theo (Conditional + Smooth Mix)", layout="centered")
st.title("YES Theo Calculator (BTC 15m Up/Down)")
st.write("Unconditional + 16-bucket conditional theo, with optional percentile-based smooth mixing.")

un = load_uncond()

cond_available = os.path.exists(COND_META_PATH)
cond_reason = ""
if cond_available:
    try:
        cm = load_cond_meta()
        missing = [fn for fn in cm["bucket_files"] if not os.path.exists(os.path.join(COND_DIR, fn))]
        if missing:
            cond_available = False
            cond_reason = f"Missing bucket files: {missing[:3]}{'...' if len(missing)>3 else ''}"
    except Exception as e:
        cond_available = False
        cond_reason = str(e)
else:
    cond_reason = f"Missing {COND_META_PATH}"

pct_available = os.path.exists(PCT_META_PATH)
pct_reason = ""
if pct_available:
    try:
        pm = json.load(open(PCT_META_PATH))
        pct_file = pm["file"]
        if not os.path.exists(pct_file):
            if os.path.exists(os.path.join("data", os.path.basename(pct_file))):
                pass
            else:
                pct_available = False
                pct_reason = f"Missing percentile file: {pct_file}"
    except Exception as e:
        pct_available = False
        pct_reason = str(e)
else:
    pct_reason = f"Missing {PCT_META_PATH}"

use_cond = st.checkbox("Use regime-conditional CDF", value=False, key="use_cond")
if use_cond and not cond_available:
    st.error("Conditional CDF requested but not available on this deployment.")
    st.write(cond_reason)
    st.stop()
if use_cond:
    c1, c2 = st.columns([1, 2])
    with c1:
        smooth_mix = st.checkbox("Smooth mix using percentile within bucket", value=True, key="smooth_mix")
    with c2:
        st.info(
            "Behind the scenes: we compute the percentile u of your past 15-minute move within its bucket "
            "using a precomputed 0.1bp lookup table. Then we compute soft weights over the previous/current/next "
            "buckets via a smooth softmax (controlled by tau and tau_c). The final CDF value is the weighted sum "
            "of those three buckets’ CDF values at the same threshold, and theo = 1 − mixed_CDF. "
            "This makes theo vary smoothly instead of jumping at bucket edges."
        )
else:
    smooth_mix = False

if smooth_mix and not pct_available:
    st.error("Smooth mix requested but percentile table not available on this deployment.")
    st.write(pct_reason)
    st.stop()

lag = st.number_input("Lag seconds remaining (1–900)", min_value=1, max_value=900, value=180, step=1, key="lag")

bucket = None
bp_past = None
if use_cond:
    bp_past = st.number_input("Past 15-minute move (bp)", value=0.0, step=5.0, format="%.1f", key="bp_past")
    bucket = bucket_id_16(float(bp_past))
    st.caption(f"Bucket = {bucket} ({BUCKET_LABELS_16[bucket]})")

mode = st.radio("Input mode", ["bps_up_now", "prices (current vs target)"], index=0, key="mode")

tau = 0.35
tau_c = 0.30
if smooth_mix:
    tau = st.number_input("tau (edge temperature)", value=0.35, step=0.05, format="%.2f", key="tau")
    tau_c = st.number_input("tau_c (center temperature)", value=0.30, step=0.05, format="%.2f", key="tau_c")

if use_cond and smooth_mix:
    u = percentile_in_bucket(float(bp_past), int(bucket))
    w_prev, w_cur, w_next = weights_softmax_3(u, float(tau), float(tau_c))

    B = len(load_cond_meta()["bucket_files"])
    b_prev = max(0, bucket - 1)
    b_cur = bucket
    b_next = min(B - 1, bucket + 1)

    def cdf_func(lag_, bp_):
        return (
            w_prev * cdf_cond(b_prev, lag_, bp_) +
            w_cur  * cdf_cond(b_cur,  lag_, bp_) +
            w_next * cdf_cond(b_next, lag_, bp_)
        )

    st.caption(f"u={u:.4f}, weights(prev,cur,next)=({w_prev:.3f},{w_cur:.3f},{w_next:.3f})")
elif use_cond:
    def cdf_func(lag_, bp_):
        return cdf_cond(bucket, lag_, bp_)
else:
    def cdf_func(lag_, bp_):
        return cdf_uncond(un, lag_, bp_)

theo_prob = float("nan")
if mode == "bps_up_now":
    bp_up_now = st.number_input("Current move vs target (bp)", value=5.0, step=0.1, format="%.1f", key="bp_up_now")
    theo_prob = theo_up_from_bps_up_now(cdf_func, int(lag), float(bp_up_now))
else:
    s_target = st.number_input("Target BTC price (start of window)", value=100000.0, step=10.0, format="%.2f", key="target")
    s_cur = st.number_input("Current BTC price", value=100050.0, step=10.0, format="%.2f", key="cur")
    theo_prob = theo_up_from_prices(cdf_func, int(lag), float(s_cur), float(s_target))

st.subheader("Result")
if math.isfinite(theo_prob):
    st.metric("theo_prob (0–1)", f"{theo_prob:.6f}")
    st.metric("theo_price (cents on $1)", f"{100.0*theo_prob:.2f}¢")
else:
    st.metric("theo_prob (0–1)", "NaN")
    st.metric("theo_price (cents on $1)", "NaN")
