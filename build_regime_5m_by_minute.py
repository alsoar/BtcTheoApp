#!/usr/bin/env python3
import math
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

PRICE_PATH = "out/close_rev.f32"  # float32, newest -> oldest
OUT_PATH = "out/regime_5m_by_minute.f32"
TMP_PATH = "out/regime_5m_metrics_tmp.f32"

WINDOW_SECONDS = 300
WINDOW_POINTS = WINDOW_SECONDS + 1
STEP_SECONDS = 60

ALPHA = 0.10
GAMMA = 0.20
CHUNK_TAGS = 25_000


def iter_tag_blocks(num_prices: int, chunk_tags: int):
    if num_prices < WINDOW_POINTS:
        return

    num_tags = ((num_prices - 1) - WINDOW_SECONDS) // STEP_SECONDS + 1
    for tag_start in range(0, num_tags, chunk_tags):
        count = min(chunk_tags, num_tags - tag_start)
        first_t = WINDOW_SECONDS + tag_start * STEP_SECONDS
        yield tag_start, count, first_t


def compute_block_metrics(price_block: np.ndarray, count: int) -> np.ndarray:
    windows = sliding_window_view(price_block, WINDOW_POINTS)[::STEP_SECONDS]
    if windows.shape[0] != count:
        raise RuntimeError(f"Expected {count} windows, got {windows.shape[0]}")

    metrics = np.full((count, 3), np.nan, dtype=np.float32)
    valid = np.all(np.isfinite(windows) & (windows > 0.0), axis=1)
    if not np.any(valid):
        return metrics

    w = windows[valid].astype(np.float64, copy=False)
    p0 = w[:, 0]
    pmax = np.max(w, axis=1)
    pmin = np.min(w, axis=1)
    log_returns = np.log(w[:, 1:] / w[:, :-1])

    rv_bp = np.round(10_000.0 * np.sqrt(np.sum(log_returns * log_returns, axis=1)), 1)
    range_bp = np.round(10_000.0 * np.log(pmax / pmin), 1)
    mid = 0.5 * (pmax + pmin)
    m_bp = np.round(10_000.0 * np.abs(np.log(mid / p0)), 1)

    metrics[valid, 0] = range_bp.astype(np.float32)
    metrics[valid, 1] = rv_bp.astype(np.float32)
    metrics[valid, 2] = m_bp.astype(np.float32)
    return metrics


def main():
    if not os.path.exists(PRICE_PATH):
        raise SystemExit(f"Missing {PRICE_PATH}")

    os.makedirs("out", exist_ok=True)

    prices_rev = np.memmap(PRICE_PATH, dtype=np.float32, mode="r")
    num_prices = prices_rev.size
    if num_prices < WINDOW_POINTS:
        raise SystemExit("Not enough prices for one 5-minute window.")

    prices = prices_rev[::-1]
    num_tags = ((num_prices - 1) - WINDOW_SECONDS) // STEP_SECONDS + 1
    print(f"N prices: {num_prices:,}")
    print(f"Tagged minutes: {num_tags:,}")

    metrics_map = np.memmap(TMP_PATH, dtype=np.float32, mode="w+", shape=(num_tags, 3))
    metrics_map[:] = np.nan

    for tag_start, count, first_t in iter_tag_blocks(num_prices, CHUNK_TAGS):
        last_t = first_t + (count - 1) * STEP_SECONDS
        price_block = prices[first_t - WINDOW_SECONDS:last_t + 1]
        metrics = compute_block_metrics(price_block, count)
        metrics_map[tag_start:tag_start + count] = metrics

        if tag_start == 0 or ((tag_start // CHUNK_TAGS) + 1) % 10 == 0:
            done = min(tag_start + count, num_tags)
            print(f"metric pass: {done:,}/{num_tags:,}")

    metrics_map.flush()

    rv_bp_all = np.asarray(metrics_map[:, 1])
    valid_mask = np.isfinite(rv_bp_all)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        raise SystemExit("No valid tagged minutes found.")

    beta_bp = float(np.median(rv_bp_all[valid_mask]))
    if not math.isfinite(beta_bp) or beta_bp <= 0.0:
        raise SystemExit(f"Invalid beta_bp={beta_bp}")

    print(f"Valid tagged minutes: {valid_count:,}")
    print(f"beta_bp (median rounded RV_bp): {beta_bp:.1f}")

    out = np.memmap(OUT_PATH, dtype=np.float32, mode="w+", shape=(valid_count, 4))

    write_pos = 0
    for tag_start in range(0, num_tags, CHUNK_TAGS):
        count = min(CHUNK_TAGS, num_tags - tag_start)
        block = np.asarray(metrics_map[tag_start:tag_start + count], dtype=np.float32)
        valid = np.isfinite(block[:, 1])
        if not np.any(valid):
            continue

        rounded = block[valid].astype(np.float64, copy=False)
        range_bp = rounded[:, 0]
        rv_bp = rounded[:, 1]
        m_bp = rounded[:, 2]
        rv_eps = np.maximum(rv_bp, 1e-6)
        score = (
            (1.0 - np.exp(-rv_bp / beta_bp))
            * np.exp(-ALPHA * range_bp / rv_eps)
            * np.exp(-GAMMA * m_bp / rv_eps)
        )

        rows = np.empty((rounded.shape[0], 4), dtype=np.float32)
        rows[:, 0] = range_bp.astype(np.float32)
        rows[:, 1] = rv_bp.astype(np.float32)
        rows[:, 2] = m_bp.astype(np.float32)
        rows[:, 3] = score.astype(np.float32)

        next_write = write_pos + rows.shape[0]
        out[write_pos:next_write] = rows
        write_pos = next_write

        if tag_start == 0 or ((tag_start // CHUNK_TAGS) + 1) % 10 == 0:
            print(f"write pass: {write_pos:,}/{valid_count:,}")

    out.flush()
    if write_pos != valid_count:
        raise RuntimeError(f"Wrote {write_pos} rows, expected {valid_count}")

    os.remove(TMP_PATH)
    print(f"DONE: {OUT_PATH}")
    print(f"Rows: {valid_count:,}")


if __name__ == "__main__":
    main()
