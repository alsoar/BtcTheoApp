import json
import math
import mmap
import os
import struct
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

def cdf_cond(bucket: int, lag: int, bp: float) -> float:
    cm = load_cond_meta()
    L, N = cm["L"], cm["N"]
    bp10_min, bp10_max = cm["bp10_min"], cm["bp10_max"]
    B = len(cm["bucket_files"])
    b = max(0, min(int(bucket), B - 1))
    lag = max(1, min(int(lag), L - 1))
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

st.set_page_config(page_title="YES Theo (Conditional + Smoothing + Plots)", layout="centered")
st.title("YES Theo Calculator (BTC 15m Up/Down)")
st.write("Includes optional smoothing and optional plots of theo as a function of a single input while holding others fixed.")

use_cond = st.checkbox("Use regime-conditional CDF", value=False, key="use_cond")

smooth_mix = False
if use_cond:
    c1, c2 = st.columns([1, 2])
    with c1:
        smooth_mix = st.checkbox("Enable smoothing", value=True, key="smooth_mix")
    with c2:
        st.info("Smoothing uses percentile u within bucket and mixes conditional CDFs; theo = 1 − mixed_CDF.")

lag_col1, lag_col2 = st.columns([4, 1])
with lag_col1:
    lag = st.number_input("Lag seconds remaining (1–900)", min_value=1, max_value=900, value=180, step=1, key="lag")
with lag_col2:
    plot_lag = st.checkbox("Plot", value=False, key="plot_lag")

bucket = None
bp_past = None
u = None

plot_bp_past = False
if use_cond:
    bp_col1, bp_col2 = st.columns([4, 1])
    with bp_col1:
        bp_past = st.number_input("Past 15-minute move (bp)", value=0.0, step=5.0, format="%.1f", key="bp_past")
    with bp_col2:
        plot_bp_past = st.checkbox("Plot", value=False, key="plot_bp_past")
    bucket = bucket_id_16(float(bp_past))
    st.caption(f"Bucket = {bucket} ({BUCKET_LABELS_16[bucket]})")

mix_mode = None
tau = 0.60
tau_c = 0.50
k = 2.0
plot_tau = False
plot_k = False

if use_cond and smooth_mix:
    mix_mode = st.radio("Smoothing mode", ["3-bucket softmax", "16-bucket smooth"], index=1, key="mix_mode")
    if mix_mode == "3-bucket softmax":
        t1, t2, t3 = st.columns([3, 3, 1])
        with t1:
            tau = st.number_input("tau (edge temperature)", value=0.60, step=0.05, format="%.2f", key="tau")
        with t2:
            tau_c = st.number_input("tau_c (center temperature)", value=0.50, step=0.05, format="%.2f", key="tau_c")
        with t3:
            plot_tau = st.checkbox("Plot", value=False, key="plot_tau")
    else:
        k1, k2c = st.columns([4, 1])
        with k1:
            k = st.number_input("k (spread across bucket index)", value=2.0, step=0.25, format="%.2f", key="k")
        with k2c:
            plot_k = st.checkbox("Plot", value=False, key="plot_k")

mode = st.radio("Input mode", ["bps_up_now", "prices (current vs target)"], index=0, key="mode")

plot_bp_up = False
plot_cur = False
plot_target = False

bp_up_now = 5.0
s_target = 100000.0
s_cur = 100050.0

if mode == "bps_up_now":
    b1, b2 = st.columns([4, 1])
    with b1:
        bp_up_now = st.number_input("Current move vs target (bp)", value=5.0, step=0.1, format="%.1f", key="bp_up_now")
    with b2:
        plot_bp_up = st.checkbox("Plot", value=False, key="plot_bp_up_now")
else:
    p1, p2, p3, p4 = st.columns([3, 1, 3, 1])
    with p1:
        s_target = st.number_input("Target BTC price (start of window)", value=100000.0, step=10.0, format="%.2f", key="target")
    with p2:
        plot_target = st.checkbox("Plot", value=False, key="plot_target")
    with p3:
        s_cur = st.number_input("Current BTC price", value=100050.0, step=10.0, format="%.2f", key="cur")
    with p4:
        plot_cur = st.checkbox("Plot", value=False, key="plot_cur")

un = None
if not use_cond:
    un = load_uncond()

def make_cdf_func(bp_past_val: float, lag_val: int, tau_val: float, tau_c_val: float, k_val: float):
    if not use_cond:
        g = un
        def f(lag_, bp_):
            return cdf_uncond(g, lag_, bp_)
        return f, {"mix": "uncond"}

    b = bucket_id_16(float(bp_past_val))

    if not smooth_mix:
        def f(lag_, bp_):
            return cdf_cond(b, lag_, bp_)
        return f, {"mix": "cond", "bucket": b}

    u0 = percentile_in_bucket(float(bp_past_val), int(b))
    B = len(load_cond_meta()["bucket_files"])

    if mix_mode == "3-bucket softmax":
        w_prev, w_cur, w_next = weights_softmax_3(u0, float(tau_val), float(tau_c_val))
        b_prev = max(0, b - 1)
        b_next = min(B - 1, b + 1)
        def f(lag_, bp_):
            return (
                w_prev * cdf_cond(b_prev, lag_, bp_) +
                w_cur  * cdf_cond(b,      lag_, bp_) +
                w_next * cdf_cond(b_next, lag_, bp_)
            )
        return f, {"mix": "3", "bucket": b, "u": u0, "w": [w_prev, w_cur, w_next], "bins": [b_prev, b, b_next]}
    else:
        tpos, w = weights_all_16(int(b), float(u0), float(k_val))
        def f(lag_, bp_):
            vals = np.array([cdf_cond(j, lag_, bp_) for j in range(B)], dtype=np.float64)
            return float(np.dot(w, vals))
        return f, {"mix": "16", "bucket": b, "u": u0, "t": tpos, "w": w}

cdf_func, mix_info = make_cdf_func(bp_past, int(lag), float(tau), float(tau_c), float(k))

theo_prob = float("nan")
if mode == "bps_up_now":
    bp_req = bp_req_from_bp_up_now(float(bp_up_now))
    theo_prob = theo_from_bp_req(cdf_func, int(lag), bp_req)
else:
    bp_req = bp_req_from_prices(float(s_cur), float(s_target))
    theo_prob = theo_from_bp_req(cdf_func, int(lag), bp_req)

st.subheader("Result")
if math.isfinite(theo_prob):
    st.metric("theo_price (cents on $1)", f"{100.0*theo_prob:.2f}¢")
else:
    st.metric("theo_price (cents on $1)", "NaN")

with st.expander("Smoothing weights widget", expanded=False):
    if mix_info.get("mix") == "3":
        st.write({"mode": "3-bucket softmax", "bucket": mix_info["bucket"], "u": mix_info["u"], "bins": mix_info["bins"], "weights": mix_info["w"], "tau": tau, "tau_c": tau_c})
        df = pd.DataFrame({"bucket": list(range(16)), "label": BUCKET_LABELS_16, "weight": [0.0]*16})
        for bi, wi in zip(mix_info["bins"], mix_info["w"]):
            df.loc[df["bucket"] == bi, "weight"] = wi
        st.dataframe(df, use_container_width=True)
    elif mix_info.get("mix") == "16":
        w = np.asarray(mix_info["w"], dtype=np.float64)
        st.write({"mode": "16-bucket smooth", "bucket": mix_info["bucket"], "u": mix_info["u"], "t": mix_info["t"], "k": k})
        df = pd.DataFrame({"bucket": list(range(16)), "label": BUCKET_LABELS_16, "weight": w})
        st.dataframe(df, use_container_width=True)
    elif mix_info.get("mix") == "cond":
        st.write({"mode": "conditional (no smoothing)", "bucket": mix_info["bucket"]})
    else:
        st.write({"mode": "unconditional"})

if plot_lag:
    xvals = np.arange(1, 901, dtype=np.int32)
    theo_cents = []
    if mode == "bps_up_now":
        bp_req0 = bp_req_from_bp_up_now(float(bp_up_now))
        for Lg in xvals:
            f, _ = make_cdf_func(bp_past, int(Lg), float(tau), float(tau_c), float(k))
            theo_cents.append(100.0 * theo_from_bp_req(f, int(Lg), bp_req0))
        plot_theo_line(xvals, theo_cents, "Lag (seconds)", "Theo vs lag (others fixed)")
    else:
        bp_req0 = bp_req_from_prices(float(s_cur), float(s_target))
        for Lg in xvals:
            f, _ = make_cdf_func(bp_past, int(Lg), float(tau), float(tau_c), float(k))
            theo_cents.append(100.0 * theo_from_bp_req(f, int(Lg), bp_req0))
        plot_theo_line(xvals, theo_cents, "Lag (seconds)", "Theo vs lag (others fixed)")

if plot_bp_past and use_cond:
    xvals = np.arange(-1000.0, 1000.0 + 0.1, 0.1, dtype=np.float64)
    theo_cents = []
    if mode == "bps_up_now":
        bp_req0 = bp_req_from_bp_up_now(float(bp_up_now))
        for z in xvals:
            f, _ = make_cdf_func(float(z), int(lag), float(tau), float(tau_c), float(k))
            theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_req0))
        plot_theo_line(xvals, theo_cents, "Past 15m move (bp)", "Theo vs past move (others fixed)")
    else:
        bp_req0 = bp_req_from_prices(float(s_cur), float(s_target))
        for z in xvals:
            f, _ = make_cdf_func(float(z), int(lag), float(tau), float(tau_c), float(k))
            theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_req0))
        plot_theo_line(xvals, theo_cents, "Past 15m move (bp)", "Theo vs past move (others fixed)")

if plot_bp_up and mode == "bps_up_now":
    xvals = np.arange(-500.0, 500.0 + 0.5, 0.5, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, int(lag), float(tau), float(tau_c), float(k))
        bp_reqx = bp_req_from_bp_up_now(float(x))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_reqx))
    plot_theo_line(xvals, theo_cents, "Current move vs target (bp)", "Theo vs bp_up_now (others fixed)")

if plot_cur and mode != "bps_up_now":
    xvals = np.linspace(float(s_cur) * 0.98, float(s_cur) * 1.02, 301, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, int(lag), float(tau), float(tau_c), float(k))
        bp_reqx = bp_req_from_prices(float(x), float(s_target))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_reqx))
    plot_theo_line(xvals, theo_cents, "Current BTC price", "Theo vs current price (others fixed)")

if plot_target and mode != "bps_up_now":
    xvals = np.linspace(float(s_target) * 0.98, float(s_target) * 1.02, 301, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, int(lag), float(tau), float(tau_c), float(k))
        bp_reqx = bp_req_from_prices(float(s_cur), float(x))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_reqx))
    plot_theo_line(xvals, theo_cents, "Target BTC price", "Theo vs target price (others fixed)")

if plot_tau and use_cond and smooth_mix and mix_mode == "3-bucket softmax":
    xvals = np.linspace(0.10, 1.50, 141, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, int(lag), float(x), float(tau_c), float(k))
        if mode == "bps_up_now":
            bp_req0 = bp_req_from_bp_up_now(float(bp_up_now))
        else:
            bp_req0 = bp_req_from_prices(float(s_cur), float(s_target))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_req0))
    plot_theo_line(xvals, theo_cents, "tau", "Theo vs tau (others fixed)")

if plot_k and use_cond and smooth_mix and mix_mode == "16-bucket smooth":
    xvals = np.linspace(0.25, 6.0, 116, dtype=np.float64)
    theo_cents = []
    for x in xvals:
        f, _ = make_cdf_func(bp_past, int(lag), float(tau), float(tau_c), float(x))
        if mode == "bps_up_now":
            bp_req0 = bp_req_from_bp_up_now(float(bp_up_now))
        else:
            bp_req0 = bp_req_from_prices(float(s_cur), float(s_target))
        theo_cents.append(100.0 * theo_from_bp_req(f, int(lag), bp_req0))
    plot_theo_line(xvals, theo_cents, "k", "Theo vs k (others fixed)")
