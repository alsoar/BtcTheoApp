#!/usr/bin/env python3
import argparse
import json
import math
import os

import numpy as np

DATA_PATH = "out/regime_5m_by_minute.f32"
OUT_DIR = "out"

CLIP_EPS = 1e-6
DEFAULT_ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward ridge regression on logit(S).")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in minutes.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward validation folds.")
    parser.add_argument("--train-frac", type=float, default=0.60, help="Initial train fraction before walk-forward folds.")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Candidate ridge alpha values.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable z-scoring of features using train stats.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write a calibration PNG with matplotlib.",
    )
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1.0 - p))


def load_dataset(path: str, horizon: int):
    raw = np.memmap(path, dtype=np.float32, mode="r")
    if raw.size % 4 != 0:
        raise SystemExit(f"{path} does not contain a whole number of 4-float rows.")

    rows = raw.reshape(-1, 4)
    n_rows = rows.shape[0]
    min_needed = 6 + horizon
    if n_rows < min_needed:
        raise SystemExit(f"Need at least {min_needed} rows for horizon={horizon}, found {n_rows}.")

    range_bp = rows[:, 0].astype(np.float64)
    rv_bp = rows[:, 1].astype(np.float64)
    s_raw = rows[:, 3].astype(np.float64)
    s = np.clip(s_raw, CLIP_EPS, 1.0 - CLIP_EPS)

    rolling6 = np.convolve(s, np.ones(6, dtype=np.float64) / 6.0, mode="valid")
    t_start = 5
    t_stop = n_rows - horizon
    count = t_stop - t_start
    if count <= 0:
        raise SystemExit("No valid samples after horizon alignment.")

    x = np.empty((count, 4), dtype=np.float64)
    x[:, 0] = range_bp[t_start:t_stop]
    x[:, 1] = rv_bp[t_start:t_stop]
    x[:, 2] = s[t_start:t_stop]
    x[:, 3] = rolling6[:count]

    s_target = s_raw[t_start + horizon:t_stop + horizon]
    y = logit(np.clip(s_target, CLIP_EPS, 1.0 - CLIP_EPS))
    finite = np.all(np.isfinite(x), axis=1) & np.isfinite(y) & np.isfinite(s_target)
    if not np.all(finite):
        x = x[finite]
        y = y[finite]
        s_target = s_target[finite]
    return x, y, s_target


def make_splits(n_samples: int, n_splits: int, train_frac: float):
    if not (0.0 < train_frac < 1.0):
        raise SystemExit("--train-frac must be in (0, 1).")
    initial_train = max(int(n_samples * train_frac), 1)
    remaining = n_samples - initial_train
    if remaining < n_splits:
        raise SystemExit(
            f"Not enough held-out samples ({remaining}) for {n_splits} folds after initial train={initial_train}."
        )

    base_fold = remaining // n_splits
    extra = remaining % n_splits
    splits = []
    val_start = initial_train
    for fold_idx in range(n_splits):
        fold_size = base_fold + (1 if fold_idx < extra else 0)
        val_end = val_start + fold_size
        splits.append((0, val_start, val_start, val_end))
        val_start = val_end
    return splits


def fit_ridge(x_train, y_train, alpha: float, standardize: bool):
    if standardize:
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std[x_std == 0.0] = 1.0
    else:
        x_mean = np.zeros(x_train.shape[1], dtype=np.float64)
        x_std = np.ones(x_train.shape[1], dtype=np.float64)

    y_mean = float(y_train.mean())
    xz = (x_train - x_mean) / x_std
    yz = y_train - y_mean

    gram = xz.T @ xz
    rhs = xz.T @ yz
    coef = np.linalg.solve(gram + alpha * np.eye(x_train.shape[1], dtype=np.float64), rhs)

    return {
        "coef": coef,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
    }


def predict_ridge(model, x):
    xz = (x - model["x_mean"]) / model["x_std"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        yhat = xz @ model["coef"] + model["y_mean"]
    return yhat


def build_calibration(s_true, s_pred, num_bins=10):
    order = np.argsort(s_pred)
    bins = np.array_split(order, num_bins)
    rows = []
    for i, idx in enumerate(bins, start=1):
        if idx.size == 0:
            continue
        rows.append(
            {
                "bin": i,
                "count": int(idx.size),
                "pred_mean": float(np.mean(s_pred[idx])),
                "true_mean": float(np.mean(s_true[idx])),
                "pred_min": float(np.min(s_pred[idx])),
                "pred_max": float(np.max(s_pred[idx])),
            }
        )
    return rows


def save_calibration_plot(rows, png_path):
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUT_DIR, ".mpl-cache"))
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        import matplotlib.pyplot as plt
    except Exception:
        return False

    pred = [r["pred_mean"] for r in rows]
    true = [r["true_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1)
    ax.plot(pred, true, marker="o", linewidth=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted S")
    ax.set_ylabel("Observed S")
    ax.set_title("Calibration")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    x, y, s_target = load_dataset(args.data, args.horizon)
    n_samples = x.shape[0]
    splits = make_splits(n_samples, args.n_splits, args.train_frac)
    standardize = not args.no_standardize

    print(f"Samples: {n_samples:,}")
    print(f"Horizon: {args.horizon} minute(s)")
    print(f"Walk-forward folds: {len(splits)}")

    cv_rows = []
    alpha_to_scores = {float(alpha): [] for alpha in args.alphas}

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        x_train = x[train_start:train_end]
        y_train = y[train_start:train_end]
        x_val = x[val_start:val_end]
        y_val = y[val_start:val_end]

        print(
            f"CV fold {fold_idx}: train={train_end - train_start:,} "
            f"val={val_end - val_start:,}"
        )

        for alpha in args.alphas:
            model = fit_ridge(x_train, y_train, float(alpha), standardize)
            yhat = predict_ridge(model, x_val)
            if not np.all(np.isfinite(yhat)):
                mse = math.inf
                mae = math.inf
            else:
                mse = float(np.mean((yhat - y_val) ** 2))
                mae = float(np.mean(np.abs(yhat - y_val)))
            alpha_to_scores[float(alpha)].append(mse)
            cv_rows.append(
                {
                    "fold": fold_idx,
                    "alpha": float(alpha),
                    "y_mse": mse,
                    "y_mae": mae,
                }
            )

    alpha_summary = []
    for alpha in args.alphas:
        alpha = float(alpha)
        mses = np.array(alpha_to_scores[alpha], dtype=np.float64)
        alpha_summary.append(
            {
                "alpha": alpha,
                "mean_y_mse": float(np.mean(mses)),
                "std_y_mse": float(np.std(mses)),
            }
        )
    alpha_summary.sort(key=lambda row: (row["mean_y_mse"], row["alpha"]))
    best_alpha = alpha_summary[0]["alpha"]
    print(f"Selected alpha: {best_alpha}")

    yhat_oos = np.empty(0, dtype=np.float64)
    s_pred_oos = np.empty(0, dtype=np.float64)
    s_true_oos = np.empty(0, dtype=np.float64)
    y_true_oos = np.empty(0, dtype=np.float64)
    fold_metrics = []

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(splits, start=1):
        model = fit_ridge(x[train_start:train_end], y[train_start:train_end], best_alpha, standardize)
        yhat = predict_ridge(model, x[val_start:val_end])
        if not np.all(np.isfinite(yhat)):
            raise RuntimeError(f"Non-finite predictions for selected alpha={best_alpha} on fold {fold_idx}")
        shat = sigmoid(yhat)
        s_true = s_target[val_start:val_end]
        y_true = y[val_start:val_end]

        yhat_oos = np.concatenate((yhat_oos, yhat))
        s_pred_oos = np.concatenate((s_pred_oos, shat))
        s_true_oos = np.concatenate((s_true_oos, s_true))
        y_true_oos = np.concatenate((y_true_oos, y_true))

        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_size": int(train_end - train_start),
                "val_size": int(val_end - val_start),
                "y_mse": float(np.mean((yhat - y_true) ** 2)),
                "y_mae": float(np.mean(np.abs(yhat - y_true))),
                "s_mse": float(np.mean((shat - s_true) ** 2)),
                "s_mae": float(np.mean(np.abs(shat - s_true))),
            }
        )

    final_model = fit_ridge(x, y, best_alpha, standardize)
    final_coefs = (final_model["coef"] / final_model["x_std"]).tolist()
    final_intercept = float(final_model["y_mean"] - np.dot(final_model["x_mean"] / final_model["x_std"], final_model["coef"]))

    calibration = build_calibration(s_true_oos, s_pred_oos, num_bins=10)

    stem = f"regime_ridge_h{args.horizon}"
    metrics_path = os.path.join(OUT_DIR, f"{stem}_metrics.json")
    calib_path = os.path.join(OUT_DIR, f"{stem}_calibration.csv")
    model_path = os.path.join(OUT_DIR, f"{stem}_model.npz")
    plot_path = os.path.join(OUT_DIR, f"{stem}_calibration.png")

    with open(calib_path, "w") as f:
        f.write("bin,count,pred_mean,true_mean,pred_min,pred_max\n")
        for row in calibration:
            f.write(
                f"{row['bin']},{row['count']},{row['pred_mean']:.10f},"
                f"{row['true_mean']:.10f},{row['pred_min']:.10f},{row['pred_max']:.10f}\n"
            )

    metrics = {
        "data_path": args.data,
        "horizon_minutes": args.horizon,
        "n_samples": int(n_samples),
        "n_splits": int(args.n_splits),
        "train_frac": float(args.train_frac),
        "standardized_features": bool(standardize),
        "candidate_alphas": [float(alpha) for alpha in args.alphas],
        "selected_alpha": float(best_alpha),
        "cv_alpha_summary": alpha_summary,
        "cv_rows": cv_rows,
        "fold_metrics": fold_metrics,
        "overall": {
            "y_mse": float(np.mean((yhat_oos - y_true_oos) ** 2)),
            "y_mae": float(np.mean(np.abs(yhat_oos - y_true_oos))),
            "s_mse": float(np.mean((s_pred_oos - s_true_oos) ** 2)),
            "s_mae": float(np.mean(np.abs(s_pred_oos - s_true_oos))),
        },
        "final_model": {
            "feature_names": ["Range_bp", "RV_bp", "S", "S_mean_6"],
            "coef_on_raw_x": final_coefs,
            "intercept": final_intercept,
            "x_mean": final_model["x_mean"].tolist(),
            "x_std": final_model["x_std"].tolist(),
            "coef_on_standardized_x": final_model["coef"].tolist(),
            "y_mean": float(final_model["y_mean"]),
        },
        "artifacts": {
            "calibration_csv": calib_path,
            "calibration_png": plot_path if args.plot else None,
            "model_npz": model_path,
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(
        model_path,
        coef=final_model["coef"],
        x_mean=final_model["x_mean"],
        x_std=final_model["x_std"],
        y_mean=np.array([final_model["y_mean"]], dtype=np.float64),
        alpha=np.array([best_alpha], dtype=np.float64),
        horizon=np.array([args.horizon], dtype=np.int64),
    )

    plot_written = False
    if args.plot:
        plot_written = save_calibration_plot(calibration, plot_path)
        if not plot_written:
            metrics["artifacts"]["calibration_png"] = None
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

    print(f"Metrics: {metrics_path}")
    print(f"Calibration CSV: {calib_path}")
    if plot_written:
        print(f"Calibration PNG: {plot_path}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
