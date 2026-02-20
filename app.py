import json
import math
import mmap
import os
import struct
import streamlit as st

GRID_PATH = "data/cdf_grid.f32"
META_PATH = "data/cdf_grid_meta.json"
STRICT_EPS_BP = 0.1  # one bucket below for strict >= approximation


@st.cache_resource
def load_grid(grid_path: str, meta_path: str):
    if not os.path.exists(grid_path):
        raise FileNotFoundError(f"Missing {grid_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    bp10_min = int(meta["bp10_min"])
    bp10_max = int(meta["bp10_max"])
    rows, cols = map(int, meta["shape"])

    fbin = open(grid_path, "rb")
    mm = mmap.mmap(fbin.fileno(), 0, access=mmap.ACCESS_READ)

    return {
        "meta": meta,
        "bp10_min": bp10_min,
        "bp10_max": bp10_max,
        "rows": rows,
        "cols": cols,
        "fbin": fbin,
        "mm": mm,
    }


def cdf_lookup(grid, lag: int, bp: float) -> float:
    rows = grid["rows"]
    cols = grid["cols"]
    bp10_min = grid["bp10_min"]
    bp10_max = grid["bp10_max"]
    mm = grid["mm"]

    if lag < 1:
        lag = 1
    if lag >= rows:
        lag = rows - 1

    bp10 = int(round(bp * 10.0))

    if bp10 < bp10_min:
        return 0.0
    if bp10 > bp10_max:
        return 1.0

    col = bp10 - bp10_min
    idx = lag * cols + col
    off = idx * 4
    return struct.unpack_from("<f", mm, off)[0]


def theo_up_from_prices(grid, lag: int, s_cur: float, s_target: float) -> float:
    if s_cur <= 0.0 or s_target <= 0.0 or (not math.isfinite(s_cur)) or (not math.isfinite(s_target)):
        return float("nan")
    bp_req = 10000.0 * (s_target / s_cur - 1.0)
    c = cdf_lookup(grid, lag, bp_req - STRICT_EPS_BP)
    theo = 1.0 - c
    return min(1.0, max(0.0, theo))


def theo_up_from_bps_up_now(grid, lag: int, bp_up_now: float) -> float:
    r_now = bp_up_now / 10000.0
    if (not math.isfinite(r_now)) or (1.0 + r_now) <= 0.0:
        return float("nan")
    bp_req = 10000.0 * ((1.0 / (1.0 + r_now)) - 1.0)
    c = cdf_lookup(grid, lag, bp_req - STRICT_EPS_BP)
    theo = 1.0 - c
    return min(1.0, max(0.0, theo))


# ---------------- UI ----------------
st.set_page_config(page_title="YES Theo Calculator (BTC 15m)", layout="centered")
st.title("YES Theo Calculator (BTC 15m Up/Down)")
st.write("Outputs theo as both probability (0–1) and price in cents on the dollar (0–100).")

try:
    grid = load_grid(GRID_PATH, META_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

st.caption(
    f"Grid loaded. rows={grid['rows']}, bp10_min={grid['bp10_min']}, bp10_max={grid['bp10_max']}"
)

lag = st.number_input(
    "Lag seconds remaining (1–900)",
    min_value=1,
    max_value=900,
    value=180,
    step=1,
    key="lag_seconds",
)

mode = st.radio(
    "Input mode",
    ["bps_up_now", "prices (current vs target)"],
    index=0,
    key="input_mode",
)

theo_prob = float("nan")

if mode == "bps_up_now":
    bp_up_now = st.number_input(
        "Current move vs target (bps). Example: +100 = current BTC is 1.00% above target.",
        value=5.0,
        step=0.1,
        format="%.1f",
        key="bp_up_now",
    )
    theo_prob = theo_up_from_bps_up_now(grid, int(lag), float(bp_up_now))
else:
    s_target = st.number_input(
        "Target BTC price (start of window)",
        value=100000.0,
        step=10.0,
        format="%.2f",
        key="target_price",
    )
    s_cur = st.number_input(
        "Current BTC price",
        value=100050.0,
        step=10.0,
        format="%.2f",
        key="current_price",
    )
    theo_prob = theo_up_from_prices(grid, int(lag), float(s_cur), float(s_target))

st.subheader("Result")

if math.isfinite(theo_prob):
    theo_cents = 100.0 * theo_prob
    st.metric("theo_prob (0–1)", f"{theo_prob:.6f}")
    st.metric("theo_price (cents on $1)", f"{theo_cents:.2f}¢")
else:
    st.metric("theo_prob (0–1)", "NaN")
    st.metric("theo_price (cents on $1)", "NaN")

st.divider()
st.caption("Theo uses: theo_up = 1 - CDF(lag, bp_req - 0.1). Price in cents = 100 * theo_up.")
