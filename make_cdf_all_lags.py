import csv
import os

INP = "out/bp_lags_1_900_counts.csv"
OUT = "out/bp_lags_1_900_cdf.csv"

os.makedirs("out", exist_ok=True)

with open(INP, newline="") as f_in, open(OUT, "w", newline="") as f_out:
    r = csv.DictReader(f_in)
    w = csv.writer(f_out)
    w.writerow(["lag_seconds", "bp_move", "cdf"])

    current_lag = None
    lag_rows = []     # list of (bp_move, count)
    lag_total = 0

    def flush_lag(lag, rows, total):
        if lag is None:
            return
        if total == 0:
            return
        # rows should already be sorted by bp_move
        cum = 0
        for bp, c in rows:
            cum += c
            w.writerow([lag, bp, cum / total])

    for row in r:
        lag = int(row["lag_seconds"])
        bp = float(row["bp_move"])
        c = int(row["count"])

        if current_lag is None:
            current_lag = lag

        if lag != current_lag:
            flush_lag(current_lag, lag_rows, lag_total)
            current_lag = lag
            lag_rows = []
            lag_total = 0

        lag_rows.append((bp, c))
        lag_total += c

    # flush last lag
    flush_lag(current_lag, lag_rows, lag_total)

print("Wrote:", OUT)
