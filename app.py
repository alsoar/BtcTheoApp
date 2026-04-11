import json
import math
import mmap
import os
import struct
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

APP_VERSION = "deploy-twap-grid-v3"

TWAP_SECONDS = 60

UN_GRID_PATH  = "data/cdf_grid.f32"
UN_META_PATH  = "data/cdf_grid_meta.json"
UN_TWAP_GRID_PATH = "data/cdf_grid_twap60.f32"
UN_TWAP_META_PATH = "data/cdf_grid_twap60_meta.json"

COND_META_PATH = "data/cond16_cdf_meta.json"
COND_TWAP_META_PATH = "data/cond16_twap60_meta.json"

PCT_META_PATH = "data/bp_past_pct_meta.json"
RAW_PRICE_PATH = "out/close_rev.f32"
RAW_TWAP_PATH = "out/close_twap60_rev.f32"
RAW_BUCKET16_PATH = "out/bucket16_id_u8.bin"

STRICT_EPS_BP = 0.1
GRID_BP_MIN = -4000.0
GRID_BP_MAX = 5000.0
GRID_BP10_MIN = int(round(GRID_BP_MIN * 10.0))
GRID_BP10_MAX = int(round(GRID_BP_MAX * 10.0))
GRID_NUM_BINS = GRID_BP10_MAX - GRID_BP10_MIN + 1
TWAP_SCAN_CHUNK = 2_000_000

@st.cache_resource
def load_uncond(grid_path: str, meta_path: str):
    m = json.load(open(meta_path))
    rows, cols = map(int, m["shape"])
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    f = open(grid_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {
        "rows": rows,
        "cols": cols,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "grid_path": grid_path,
        "meta_path": meta_path,
        "f": f,
        "mm": mm,
    }

@st.cache_resource
def load_cond_meta(meta_path: str):
    m = json.load(open(meta_path))
    L, N = map(int, m["bucket_file_shape"])
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    bucket_files = m["bucket_files"]
    base_dir = os.path.dirname(meta_path) or "."
    return {
        "L": L,
        "N": N,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "bucket_files": bucket_files,
        "base_dir": base_dir,
        "meta_path": meta_path,
    }

@st.cache_resource
def load_bucket_file(meta_path: str, bucket: int):
    cm = load_cond_meta(meta_path)
    B = len(cm["bucket_files"])
    b = max(0, min(int(bucket), B - 1))
    fname = cm["bucket_files"][b]
    path = os.path.join(cm["base_dir"], fname)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"path": path, "f": f, "mm": mm}

def cond_grid_available(meta_path: str) -> bool:
    if not os.path.exists(meta_path):
        return False
    try:
        m = json.load(open(meta_path))
        base_dir = os.path.dirname(meta_path) or "."
        bucket_files = m.get("bucket_files", [])
        return bool(bucket_files) and all(os.path.exists(os.path.join(base_dir, fname)) for fname in bucket_files)
    except Exception:
        return False


@st.cache_resource
def load_raw_prices():
    return np.memmap(RAW_PRICE_PATH, dtype=np.float32, mode="r")


@st.cache_resource
def load_raw_twap():
    return np.memmap(RAW_TWAP_PATH, dtype=np.float32, mode="r")


@st.cache_resource
def load_raw_bucket16():
    return np.memmap(RAW_BUCKET16_PATH, dtype=np.uint8, mode="r")


@st.cache_resource
def compute_twap_uncond_row(lag: int) -> np.ndarray:
    lag = int(lag)
    x = load_raw_prices()
    twap = load_raw_twap()
    n = x.size
    end = min(n, twap.size + lag)
    counts = np.zeros(GRID_NUM_BINS, dtype=np.int64)

    for i0 in range(lag, end, TWAP_SCAN_CHUNK):
        i1 = min(i0 + TWAP_SCAN_CHUNK, end)
        cur = x[i0:i1].astype(np.float64, copy=False)
        avg = twap[i0 - lag:i1 - lag].astype(np.float64, copy=False)
        mask = (cur > 0.0) & np.isfinite(cur) & np.isfinite(avg) & (avg > 0.0)
        if not np.any(mask):
            continue
        bp = 10000.0 * (avg[mask] / cur[mask] - 1.0)
        bp10 = np.rint(bp * 10.0).astype(np.int32)
        bp10 = np.clip(bp10, GRID_BP10_MIN, GRID_BP10_MAX)
        counts += np.bincount(bp10 - GRID_BP10_MIN, minlength=GRID_NUM_BINS)

    total = counts.sum()
    if total == 0:
        return np.full(GRID_NUM_BINS, np.nan, dtype=np.float32)
    return (np.cumsum(counts, dtype=np.float64) / float(total)).astype(np.float32)


@st.cache_resource
def compute_twap_cond_rows(lag: int) -> np.ndarray:
    lag = int(lag)
    x = load_raw_prices()
    twap = load_raw_twap()
    bmap = load_raw_bucket16()
    n = x.size
    start = max(900, lag)
    end = min(n - 900, twap.size + lag)
    counts_flat = np.zeros(16 * GRID_NUM_BINS, dtype=np.int64)

    for i0 in range(start, end, TWAP_SCAN_CHUNK):
        i1 = min(i0 + TWAP_SCAN_CHUNK, end)
        cur = x[i0:i1].astype(np.float64, copy=False)
        avg = twap[i0 - lag:i1 - lag].astype(np.float64, copy=False)
        bb = bmap[i0:i1].astype(np.int32, copy=False)
        mask = (bb != 255) & (cur > 0.0) & np.isfinite(cur) & np.isfinite(avg) & (avg > 0.0)
        if not np.any(mask):
            continue
        bp = 10000.0 * (avg[mask] / cur[mask] - 1.0)
        bp10 = np.rint(bp * 10.0).astype(np.int32)
        bp10 = np.clip(bp10, GRID_BP10_MIN, GRID_BP10_MAX)
        key = bb[mask] * GRID_NUM_BINS + (bp10 - GRID_BP10_MIN)
        counts_flat += np.bincount(key, minlength=16 * GRID_NUM_BINS)

    counts = counts_flat.reshape(16, GRID_NUM_BINS)
    rows = np.full((16, GRID_NUM_BINS), np.nan, dtype=np.float32)
    for bi in range(16):
        total = counts[bi].sum()
        if total > 0:
            rows[bi, :] = (np.cumsum(counts[bi], dtype=np.float64) / float(total)).astype(np.float32)
    return rows

@st.cache_resource
def load_pct():
    m = json.load(open(PCT_META_PATH))
    file_path = m["file"]
    if not os.path.isabs(file_path):
        alt = os.path.join("data", os.path.basename(file_path))
        if os.path.exists(alt):
            file_path = alt
    bp10_min = int(m["bp10_min"])
    bp10_max = int(m["bp10_max"])
    nbins = int(m["nbins"])
    f = open(file_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return {"bp10_min": bp10_min, "bp10_max": bp10_max, "nbins": nbins, "file_path": file_path, "f": f, "mm": mm}

def cdf_uncond(g, lag: int, bp: float) -> float:
    rows, cols = g["rows"], g["cols"]
    bp10_min, bp10_max = g["bp10_min"], g["bp10_max"]
    mm = g["mm"]
    lag = max(1, min(int(lag), rows - 1))
    bp10 = int(round(bp * 10.0))
    if bp10 < bp10_min: return 0.0
    if bp10 > bp10_max: return 1.0
    col = bp10 - bp10_min
    idx = lag * cols + col
    return struct.unpack_from("<f", mm, idx * 4)[0]

def cdf_cond(meta_path: str, bucket: int, lag: int, bp: float) -> float:
    cm = load_cond_meta(meta_path)
    L, N = cm["L"], cm["N"]
    bp10_min, bp10_max = cm["bp10_min"], cm["bp10_max"]
    B = len(cm["bucket_files"])
    b = max(0, min(int(bucket), B - 1))
    lag = max(1, min(int(lag), L - 1))
    bp10 = int(round(bp * 10.0))
    if bp10 < bp10_min: return 0.0
    if bp10 > bp10_max: return 1.0
    col = bp10 - bp10_min
    bf = load_bucket_file(meta_path, b)
    idx = lag * N + col
    return struct.unpack_from("<f", bf["mm"], idx * 4)[0]


def cdf_row_lookup(row: np.ndarray, bp: float) -> float:
    bp10 = int(round(bp * 10.0))
    if bp10 < GRID_BP10_MIN:
        return 0.0
    if bp10 > GRID_BP10_MAX:
        return 1.0
    return float(row[bp10 - GRID_BP10_MIN])

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
    b = max(0, min(int(bucket), 15))
    off = (b * nbins + col) * 2
    u = struct.unpack_from("<e", p["mm"], off)[0]
    u = float(u) if math.isfinite(float(u)) else 0.5
    if u < 0.0: u = 0.0
    if u > 1.0: u = 1.0
    return u

def weights_softmax_3(u: float, tau: float, tau_c: float):
    dL = u
    dU = 1.0 - u
    dC = abs(u - 0.5)
    sL = -dL / max(1e-12, float(tau))
    sU = -dU / max(1e-12, float(tau))
    sC = -dC / max(1e-12, float(tau_c))
    m = max(sL, sC, sU)
    eL = math.exp(sL - m)
    eC = math.exp(sC - m)
    eU = math.exp(sU - m)
    Z = eL + eC + eU
    return eL / Z, eC / Z, eU / Z

def weights_all_16(b: int, u: float, k: float):
    t = float(b) + (float(u) - 0.5)
    j = np.arange(16, dtype=np.float64)
    scores = -np.abs(j - t) / max(1e-12, float(k))
    m = float(scores.max())
    w = np.exp(scores - m)
    w = w / float(w.sum())
    return float(t), w

def bp_req_from_bp_up_now(bp_up_now: float) -> float:
    r_now = float(bp_up_now) / 10000.0
    if (not math.isfinite(r_now)) or (1.0 + r_now) <= 0.0:
        return float("nan")
    return 10000.0 * ((1.0 / (1.0 + r_now)) - 1.0)

def bp_req_from_prices(s_cur: float, s_target: float) -> float:
    s_cur = float(s_cur)
    s_target = float(s_target)
    if s_cur <= 0.0 or s_target <= 0.0 or (not math.isfinite(s_cur)) or (not math.isfinite(s_target)):
        return float("nan")
    return 10000.0 * (s_target / s_cur - 1.0)

def theo_from_bp_req(cdf_func, lag: int, bp_req: float) -> float:
    if not math.isfinite(float(bp_req)):
        return float("nan")
    c = cdf_func(int(lag), float(bp_req) - STRICT_EPS_BP)
    t = 1.0 - float(c)
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    return t

def plot_theo_line(xvals, theo_cents, xlabel, title):
    xvals = np.asarray(xvals, dtype=np.float64)
    theo_cents = np.asarray(theo_cents, dtype=np.float64)
    fig = plt.figure(figsize=(12, 5))
    above = theo_cents >= 50.0
    idx = np.where(above[:-1] != above[1:])[0] + 1
    segments = np.split(np.arange(len(xvals)), idx)
    for seg in segments:
        if len(seg) == 0:
            continue
        color = "green" if theo_cents[seg[0]] >= 50.0 else "red"
        plt.plot(xvals[seg], theo_cents[seg], color=color)
    plt.axhline(50.0, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel("YES theo (cents on $1)")
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

st.set_page_config(page_title="YES Theo (Smoothing + Plots)", layout="centered")
st.title("YES Theo Calculator (BTC 15m Up/Down)")
st.caption(f"version: {APP_VERSION}")

use_cond = st.checkbox("Use regime-conditional CDF", value=False, key="use_cond")

smooth_mix = False
mix_mode = "16-bucket smooth"
tau = 0.60
tau_c = 0.50
k = 2.0

if use_cond:
    smooth_mix = st.checkbox("Enable smoothing", value=True, key="smooth_mix")
    if smooth_mix:
        mix_mode = st.radio("Smoothing mode", ["3-bucket softmax", "16-bucket smooth"], index=1, key="mix_mode")
        if mix_mode == "3-bucket softmax":
            tau = st.number_input("tau (edge temperature)", value=0.60, step=0.05, format="%.2f", key="tau")
            tau_c = st.number_input("tau_c (center temperature)", value=0.50, step=0.05, format="%.2f", key="tau_c")
        else:
            k = st.number_input("k (spread across bucket index)", value=2.0, step=0.25, format="%.2f", key="k")

lag = st.number_input("Lag seconds remaining (1–900)", min_value=1, max_value=900, value=180, step=1, key="lag")
twap_uncond_available = os.path.exists(UN_TWAP_GRID_PATH) and os.path.exists(UN_TWAP_META_PATH)
twap_cond_available = cond_grid_available(COND_TWAP_META_PATH)
twap_grid_available = twap_uncond_available and ((not use_cond) or twap_cond_available)
twap_fallback_available = (
    os.path.exists(RAW_PRICE_PATH)
    and os.path.exists(RAW_TWAP_PATH)
    and ((not use_cond) or os.path.exists(RAW_BUCKET16_PATH))
)
twap_available = twap_grid_available or twap_fallback_available
use_twap = st.checkbox("Use 60-point TWAP resolution", value=False, key="use_twap", disabled=(not twap_available))
if not twap_available:
    if use_cond:
        st.caption("TWAP toggle requires either the TWAP conditional grid files or `out/close_twap60_rev.f32` plus `out/bucket16_id_u8.bin`.")
    else:
        st.caption("TWAP toggle requires either the TWAP grid files or `out/close_twap60_rev.f32` for on-demand row building.")

twap_runtime_mode = "spot"
if use_twap:
    twap_runtime_mode = "grid" if twap_grid_available else "on_demand"

active_un_grid_path = UN_TWAP_GRID_PATH if use_twap else UN_GRID_PATH
active_un_meta_path = UN_TWAP_META_PATH if use_twap else UN_META_PATH
active_cond_meta_path = COND_TWAP_META_PATH if use_twap else COND_META_PATH

bp_past = 0.0
bucket = None
if use_cond:
    bp_past = st.number_input("Past 15-minute move (bp)", value=0.0, step=5.0, format="%.1f", key="bp_past")
    bucket = bucket_id_16(float(bp_past))
    st.caption(f"Bucket = {bucket} ({BUCKET_LABELS_16[bucket]})")

mode = st.radio("Input mode", ["bps_up_now", "prices (current vs target)"], index=0, key="mode")

bp_up_now = 5.0
s_target = 100000.0
s_cur = 100050.0

if mode == "bps_up_now":
    bp_up_now = st.number_input("Current move vs target (bp)", value=5.0, step=0.1, format="%.1f", key="bp_up_now")
else:
    s_target = st.number_input("Target BTC price (start of window)", value=100000.0, step=10.0, format="%.2f", key="target")
    s_cur = st.number_input("Current BTC price", value=100050.0, step=10.0, format="%.2f", key="cur")

un = None
if not use_cond and (not use_twap or twap_runtime_mode == "grid"):
    un = load_uncond(active_un_grid_path, active_un_meta_path)

def make_cdf_func(bp_past_val: float, tau_val: float, tau_c_val: float, k_val: float):
    if use_twap and twap_runtime_mode == "on_demand":
        if not use_cond:
            row = compute_twap_uncond_row(int(lag))
            def f(lag_, bp_):
                return cdf_row_lookup(row, bp_)
            return f, {"mix": "uncond", "source": "twap60-on-demand", "lag": int(lag)}

        rows = compute_twap_cond_rows(int(lag))
        b = bucket_id_16(float(bp_past_val))

        if not smooth_mix:
            def f(lag_, bp_):
                return cdf_row_lookup(rows[b], bp_)
            return f, {"mix": "cond", "bucket": b, "source": "twap60-on-demand", "lag": int(lag)}

        u0 = percentile_in_bucket(float(bp_past_val), int(b))
        b_count = rows.shape[0]

        if mix_mode == "3-bucket softmax":
            w_prev, w_cur, w_next = weights_softmax_3(u0, float(tau_val), float(tau_c_val))
            b_prev = max(0, b - 1)
            b_next = min(b_count - 1, b + 1)
            def f(lag_, bp_):
                return (
                    w_prev * cdf_row_lookup(rows[b_prev], bp_) +
                    w_cur  * cdf_row_lookup(rows[b],      bp_) +
                    w_next * cdf_row_lookup(rows[b_next], bp_)
                )
            weights_vec = np.zeros(b_count, dtype=np.float64)
            weights_vec[b_prev] += w_prev
            weights_vec[b] += w_cur
            weights_vec[b_next] += w_next
            return f, {
                "mix": "3",
                "bucket": b,
                "u": u0,
                "weights": weights_vec,
                "params": {"tau": float(tau_val), "tau_c": float(tau_c_val)},
                "source": "twap60-on-demand",
                "lag": int(lag),
            }

        tpos, w = weights_all_16(int(b), float(u0), float(k_val))
        def f(lag_, bp_):
            vals = np.array([cdf_row_lookup(rows[j], bp_) for j in range(b_count)], dtype=np.float64)
            return float(np.dot(w, vals))
        return f, {
            "mix": "16",
            "bucket": b,
            "u": u0,
            "t": tpos,
            "weights": w,
            "params": {"k": float(k_val)},
            "source": "twap60-on-demand",
            "lag": int(lag),
        }

    if not use_cond:
        g = un
        def f(lag_, bp_):
            return cdf_uncond(g, lag_, bp_)
        return f, {"mix": "uncond", "source": "twap60" if use_twap else "spot"}

    b = bucket_id_16(float(bp_past_val))
    B = len(load_cond_meta(active_cond_meta_path)["bucket_files"])

    if not smooth_mix:
        def f(lag_, bp_):
            return cdf_cond(active_cond_meta_path, b, lag_, bp_)
        return f, {"mix": "cond", "bucket": b, "source": "twap60" if use_twap else "spot"}

    u0 = percentile_in_bucket(float(bp_past_val), int(b))

    if mix_mode == "3-bucket softmax":
        w_prev, w_cur, w_next = weights_softmax_3(u0, float(tau_val), float(tau_c_val))
        b_prev = max(0, b - 1)
        b_next = min(B - 1, b + 1)
        def f(lag_, bp_):
            return (
                w_prev * cdf_cond(active_cond_meta_path, b_prev, lag_, bp_) +
                w_cur  * cdf_cond(active_cond_meta_path, b,      lag_, bp_) +
                w_next * cdf_cond(active_cond_meta_path, b_next, lag_, bp_)
            )
        weights_vec = np.zeros(B, dtype=np.float64)
        weights_vec[b_prev] += w_prev
        weights_vec[b] += w_cur
        weights_vec[b_next] += w_next
        return f, {"mix": "3", "bucket": b, "u": u0, "weights": weights_vec, "params": {"tau": float(tau_val), "tau_c": float(tau_c_val)}, "source": "twap60" if use_twap else "spot"}
    else:
        tpos, w = weights_all_16(int(b), float(u0), float(k_val))
        def f(lag_, bp_):
            vals = np.array([cdf_cond(active_cond_meta_path, j, lag_, bp_) for j in range(B)], dtype=np.float64)
            return float(np.dot(w, vals))
        return f, {"mix": "16", "bucket": b, "u": u0, "t": tpos, "weights": w, "params": {"k": float(k_val)}, "source": "twap60" if use_twap else "spot"}

cdf_func, mix_info = make_cdf_func(bp_past, float(tau), float(tau_c), float(k))

bp_req = bp_req_from_bp_up_now(float(bp_up_now)) if mode == "bps_up_now" else bp_req_from_prices(float(s_cur), float(s_target))
theo_prob = theo_from_bp_req(cdf_func, int(lag), bp_req)

st.subheader("Result")
if math.isfinite(theo_prob):
    st.metric("theo_price (cents on $1)", f"{100.0*theo_prob:.2f}¢")
else:
    st.metric("theo_price (cents on $1)", "NaN")

if use_twap:
    if twap_runtime_mode == "grid":
        st.caption(
            f"TWAP mode uses dedicated CDF grids built from the {TWAP_SECONDS} one-second BTC prices ending at resolution, "
            f"then computes bp from that average price versus the current BTC price."
        )
    else:
        st.caption(
            f"TWAP mode is computing the exact {TWAP_SECONDS}-point TWAP CDF for the selected lag on demand from `out/close_rev.f32`."
        )

with st.expander("Smoothing weights widget", expanded=False):
    st.write(mix_info)
    if mix_info.get("mix") in ("3","16"):
        w = np.asarray(mix_info["weights"], dtype=np.float64)
        df = pd.DataFrame({"bucket": list(range(16)), "label": BUCKET_LABELS_16, "weight": w})
        st.dataframe(df, use_container_width=True)

with st.sidebar:
    st.header("Plots")
    st.caption("Plot theo as a function of one input while holding others fixed.")
    if use_twap and twap_runtime_mode == "on_demand":
        st.caption("`Theo vs lag` is disabled in on-demand TWAP mode because only the selected lag row is computed.")
        plot_lag = False
    else:
        plot_lag = st.checkbox("Theo vs lag", value=False, key="plot_lag")
    plot_bp_past = st.checkbox("Theo vs past 15m move", value=False, key="plot_bp_past") if use_cond else False
    plot_bp_up = st.checkbox("Theo vs current move (bp_up_now)", value=False, key="plot_bp_up") if mode == "bps_up_now" else False
    plot_cur = st.checkbox("Theo vs current BTC price", value=False, key="plot_cur") if mode != "bps_up_now" else False
    plot_target = st.checkbox("Theo vs target BTC price", value=False, key="plot_target") if mode != "bps_up_now" else False
    plot_tau = st.checkbox("Theo vs tau", value=False, key="plot_tau") if (use_cond and smooth_mix and mix_mode=="3-bucket softmax") else False
    plot_k = st.checkbox("Theo vs k", value=False, key="plot_k") if (use_cond and smooth_mix and mix_mode=="16-bucket smooth") else False

if plot_lag:
    xvals = np.arange(1, 901, dtype=np.int32)
    theo_cents = []
    for Lg in xvals:
        f, _ = make_cdf_func(bp_past, float(tau), float(tau_c), float(k))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(Lg), bp_req))
    plot_theo_line(xvals, theo_cents, "Lag (seconds)", "Theo vs lag (others fixed)")

if plot_bp_past and use_cond:
    xvals = np.arange(-1000.0, 1000.0 + 0.1, 0.1, dtype=np.float64)
    theo_cents = []
    for z in xvals:
        f, _ = make_cdf_func(float(z), float(tau), float(tau_c), float(k))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_req))
    plot_theo_line(xvals, theo_cents, "Past 15m move (bp)", "Theo vs past move (others fixed)")

if plot_bp_up and mode == "bps_up_now":
    xvals = np.arange(-500.0, 500.0 + 0.5, 0.5, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, float(tau), float(tau_c), float(k))
        bp_reqx = bp_req_from_bp_up_now(float(x))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_reqx))
    plot_theo_line(xvals, theo_cents, "Current move vs target (bp)", "Theo vs bp_up_now (others fixed)")

if plot_cur and mode != "bps_up_now":
    xvals = np.linspace(float(s_cur) * 0.98, float(s_cur) * 1.02, 301, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, float(tau), float(tau_c), float(k))
        bp_reqx = bp_req_from_prices(float(x), float(s_target))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_reqx))
    plot_theo_line(xvals, theo_cents, "Current BTC price", "Theo vs current price (others fixed)")

if plot_target and mode != "bps_up_now":
    xvals = np.linspace(float(s_target) * 0.98, float(s_target) * 1.02, 301, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, float(tau), float(tau_c), float(k))
        bp_reqx = bp_req_from_prices(float(s_cur), float(x))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_reqx))
    plot_theo_line(xvals, theo_cents, "Target BTC price", "Theo vs target price (others fixed)")

if plot_tau and use_cond and smooth_mix and mix_mode == "3-bucket softmax":
    xvals = np.linspace(0.10, 1.50, 141, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, float(x), float(tau_c), float(k))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_req))
    plot_theo_line(xvals, theo_cents, "tau", "Theo vs tau (others fixed)")

if plot_k and use_cond and smooth_mix and mix_mode == "16-bucket smooth":
    xvals = np.linspace(0.25, 6.0, 116, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, float(tau), float(tau_c), float(x))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_req))
    plot_theo_line(xvals, theo_cents, "k", "Theo vs k (others fixed)")
