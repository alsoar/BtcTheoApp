import json
import math
import mmap
import os
import struct
import streamlit as st

# ---------- Paths ----------
UN_GRID_PATH  = "data/cdf_grid.f32"
UN_META_PATH  = "data/cdf_grid_meta.json"

COND_GRID_PATH = "data/cond_cdf_grid.f32"
COND_META_PATH = "data/cond_cdf_meta.json"

STRICT_EPS_BP = 0.1  # one 0.1bp bucket below for strict >= approximation


# ---------- mmap loader ----------
@st.cache_resource
def load_uncond():
    m = json.load(open(UN_META_PATH))
    rows, cols = map(int, m["shape"])
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    f = open(UN_GRID_PATH, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"meta": m, "rows": rows, "cols": cols, "bp10_min": bp10_min, "bp10_max": bp10_max, "f": f, "mm": mm}

@st.cache_resource
def load_cond():
    m = json.load(open(COND_META_PATH))
    B, L, N = m["shape"]  # [10, 901, num_bins]
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    f = open(COND_GRID_PATH, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"meta": m, "B": int(B), "L": int(L), "N": int(N), "bp10_min": bp10_min, "bp10_max": bp10_max, "f": f, "mm": mm}

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

def cdf_cond(g, bucket: int, lag: int, bp: float) -> float:
    B, L, N = g["B"], g["L"], g["N"]
    bp10_min, bp10_max = g["bp10_min"], g["bp10_max"]
    mm = g["mm"]

    bucket = max(0, min(bucket, B - 1))
    lag = max(1, min(lag, L - 1))
    bp10 = int(round(bp * 10.0))
    if bp10 < bp10_min: return 0.0
    if bp10 > bp10_max: return 1.0
    col = bp10 - bp10_min
    idx = ((bucket * L + lag) * N + col)
    return struct.unpack_from("<f", mm, idx * 4)[0]

# ---------- Buckets (past 15m move) ----------
def bucket_id(bp_past: float) -> int:
    if bp_past <= -800.0: return 0
    if bp_past <= -400.0: return 1
    if bp_past <= -200.0: return 2
    if bp_past <= -100.0: return 3
    if bp_past <= 0.0:    return 4
    if bp_past <= 100.0:  return 5
    if bp_past <= 200.0:  return 6
    if bp_past <= 400.0:  return 7
    if bp_past < 800.0:   return 8
    return 9

BUCKET_LABELS = [
    "≤ -800",
    "(-800, -400]",
    "(-400, -200]",
    "(-200, -100]",
    "(-100, 0]",
    "(0, 100]",
    "(100, 200]",
    "(200, 400]",
    "(400, 800)",
    "≥ 800",
]

# ---------- Theo ----------
def theo_up_from_prices(cdf_func, lag: int, s_cur: float, s_target: float) -> float:
    if s_cur <= 0.0 or s_target <= 0.0 or (not math.isfinite(s_cur)) or (not math.isfinite(s_target)):
        return float("nan")
    bp_req = 10000.0 * (s_target / s_cur - 1.0)
    c = cdf_func(lag, bp_req - STRICT_EPS_BP)
    t = 1.0 - c
    return min(1.0, max(0.0, t))

def theo_up_from_bps_up_now(cdf_func, lag: int, bp_up_now: float) -> float:
    r_now = bp_up_now / 10000.0
    if (not math.isfinite(r_now)) or (1.0 + r_now) <= 0.0:
        return float("nan")
    bp_req = 10000.0 * ((1.0 / (1.0 + r_now)) - 1.0)
    c = cdf_func(lag, bp_req - STRICT_EPS_BP)
    t = 1.0 - c
    return min(1.0, max(0.0, t))


# ---------- UI ----------
st.set_page_config(page_title="YES Theo (Unconditional + Regime Conditional)", layout="centered")
st.title("YES Theo Calculator (BTC 15m Up/Down)")
st.write("Now supports regime-conditional CDFs based on the **past 15-minute move**.")

# Load grids
try:
    un = load_uncond()
except Exception as e:
    st.error(f"Failed loading unconditional grid: {e}")
    st.stop()

cond_available = True
try:
    cd = load_cond()
except Exception:
    cond_available = False
    cd = None

use_cond = st.checkbox("Use regime-conditional CDF", value=cond_available, disabled=(not cond_available))

lag = st.number_input("Lag seconds remaining (1–900)", min_value=1, max_value=900, value=180, step=1, key="lag")

bucket = None
if use_cond:
    bp_past = st.number_input("Past 15-minute move (bp)", value=0.0, step=10.0, format="%.1f", key="bp_past")
    bucket = bucket_id(float(bp_past))
    st.caption(f"Bucket = {bucket}  ({BUCKET_LABELS[bucket]})")

mode = st.radio("Input mode", ["bps_up_now", "prices (current vs target)"], index=0, key="mode")

# Choose CDF function
if use_cond:
    def cdf_func(lag_, bp_):
        return cdf_cond(cd, bucket, lag_, bp_)
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

if use_cond:
    st.caption("Theo computed from conditional CDF: bucketed by past 15-minute move.")
else:
    st.caption("Theo computed from unconditional CDF.")
