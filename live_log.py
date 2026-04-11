#!/usr/bin/env python3
import time, json, csv, os
import urllib.request, urllib.parse
import mmap, struct
from datetime import datetime, timezone

# ---------------- CONFIG ----------------
CLOB_HOST  = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"

# Your prebuilt CDF grid files
CDF_GRID_PATH = "out/cdf_grid.f32"
CDF_META_PATH = "out/cdf_grid_meta.json"

# Output log
LOG_PATH = "out/paper_live_log.csv"

# Polling interval
DT = 1.0

# For strict >= handling: query bp_req - 0.1 bp
STRICT_EPS_BP = 0.1

# Entry/exit backtest will use these columns:
# ts, window_start, seconds_left, slug, btc_cur, btc_target, up_bid, up_ask, theo_up

# ---------------- HTTP helpers (with retry) ----------------
def http_get_json(url: str, timeout=6, retries=6, base_sleep=0.25):
    req = urllib.request.Request(url, headers={"User-Agent": "paper-log/1.0"})
    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** i))
    raise last_err

# ---------------- Coinbase BTC-USD ticker (public) ----------------
def btc_usd_coinbase():
    j = http_get_json("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5, retries=4)
    return float(j["price"])

# ---------------- Gamma helpers ----------------
def iso_to_ts(iso_str: str) -> int:
    # Example: "2026-02-17T18:30:00Z"
    return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())

def gamma_events_newest(limit=250):
    # Pull newest, open events first. This is robust to your local clock / flooring mismatch.
    q = urllib.parse.urlencode({
        "order": "id",
        "ascending": "false",
        "closed": "false",
        "limit": str(limit),
        "offset": "0",
    })
    return http_get_json(f"{GAMMA_HOST}/events?{q}", timeout=10, retries=6)

def find_active_btc_15m_slug(now_ts: int):
    """
    Returns the slug for the currently active BTC 15m up/down event,
    based on startDate/endDate containing now_ts.
    """
    events = gamma_events_newest(limit=250)

    best_slug = None
    best_end = None

    for ev in events:
        slug = (ev.get("slug") or "").lower()

        # Slug pattern in your example: btc-updown-15m-<start_ts>
        if not slug.startswith("btc-updown-15m-"):
            continue

        # event time bounds
        start_iso = ev.get("startDate") or ev.get("startDateIso")
        end_iso   = ev.get("endDate")   or ev.get("endDateIso")
        if not start_iso or not end_iso:
            continue

        try:
            start_ts = iso_to_ts(start_iso)
            end_ts = iso_to_ts(end_iso)
        except Exception:
            continue

        # must be active now
        if start_ts <= now_ts < end_ts:
            # If multiple match, choose the one ending soonest (most “current”)
            if best_slug is None or end_ts < best_end:
                best_slug = slug
                best_end = end_ts

    return best_slug  # None if not found

def gamma_market_by_slug(slug: str):
    q = urllib.parse.urlencode({"slug": slug})
    arr = http_get_json(f"{GAMMA_HOST}/markets?{q}", timeout=10, retries=6)
    return arr[0] if isinstance(arr, list) and arr else None

def parse_list_field(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return v
        except:
            pass
    return None

def get_up_token_id(slug: str):
    """
    From Gamma market metadata, map outcomes -> clobTokenIds and pick outcome == "Up".
    """
    m = gamma_market_by_slug(slug)
    if not m:
        return None

    outcomes = parse_list_field(m.get("outcomes"))
    token_ids = parse_list_field(m.get("clobTokenIds"))

    if not outcomes or not token_ids or len(outcomes) != len(token_ids):
        return None

    for o, tid in zip(outcomes, token_ids):
        if isinstance(o, str) and o.strip().lower() == "up":
            return tid

    return None

# ---------------- CLOB helpers ----------------
def clob_book(token_id: str):
    q = urllib.parse.urlencode({"token_id": token_id})
    return http_get_json(f"{CLOB_HOST}/book?{q}", timeout=8, retries=6)

def best_bid_ask(book_json):
    bids = book_json.get("bids", [])
    asks = book_json.get("asks", [])
    bid = float(bids[0]["price"]) if bids else float("nan")
    ask = float(asks[0]["price"]) if asks else float("nan")
    return bid, ask

# ---------------- CDF Grid (RAM-safe mmap, no pandas/numpy) ----------------
class CDFGrid:
    def __init__(self, grid_path, meta_path):
        with open(meta_path, "r") as f:
            m = json.load(f)
        self.bp10_min = int(m["bp10_min"])
        self.bp10_max = int(m["bp10_max"])
        self.rows, self.cols = map(int, m["shape"])

        self.f = open(grid_path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        try:
            self.mm.close()
        except:
            pass
        try:
            self.f.close()
        except:
            pass

    def cdf(self, lag: int, bp: float) -> float:
        # clamp lag to valid rows
        if lag < 1:
            lag = 1
        if lag >= self.rows:
            lag = self.rows - 1

        bp10 = int(round(bp * 10.0))
        if bp10 < self.bp10_min:
            return 0.0
        if bp10 > self.bp10_max:
            return 1.0

        col = bp10 - self.bp10_min  # 0..cols-1
        off = (lag * self.cols + col) * 4
        return struct.unpack_from("<f", self.mm, off)[0]

def theo_up(grid: CDFGrid, lag_left: int, s_cur: float, s_target: float) -> float:
    # bp_req = 10000*(target/current - 1)
    if s_cur <= 0.0 or s_target <= 0.0:
        return float("nan")
    bp_req = 10000.0 * (s_target / s_cur - 1.0)
    c = grid.cdf(lag_left, bp_req - STRICT_EPS_BP)
    t = 1.0 - c
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t

# ---------------- Logging ----------------
def ensure_log_header():
    os.makedirs("out", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts","window_start","seconds_left","slug",
                "btc_cur","btc_target",
                "up_bid","up_ask",
                "theo_up"
            ])

# ---------------- Main ----------------
def main():
    ensure_log_header()

    if not os.path.exists(CDF_GRID_PATH) or not os.path.exists(CDF_META_PATH):
        print("Missing grid files. Expected:")
        print("  ", CDF_GRID_PATH)
        print("  ", CDF_META_PATH)
        return

    grid = CDFGrid(CDF_GRID_PATH, CDF_META_PATH)

    current_slug = None
    up_token = None
    btc_target = None
    window_start_ts = None

    print("Starting live logger (bid/ask + theo)...")

    try:
        while True:
            now = time.time()
            now_ts = int(now)

            # 1) Find active market slug from Gamma
            try:
                active_slug = find_active_btc_15m_slug(now_ts)
            except Exception as e:
                print("Gamma lookup failed:", e)
                time.sleep(0.5)
                continue

            if active_slug is None:
                print("No active BTC 15m market found yet (Gamma lag). Retrying...")
                time.sleep(0.5)
                continue

            # 2) If market changed, reset state
            if active_slug != current_slug:
                current_slug = active_slug
                up_token = None
                btc_target = None

                # parse window_start from slug suffix
                try:
                    window_start_ts = int(current_slug.split("-")[-1])
                except Exception:
                    window_start_ts = None

                print("\n--- ACTIVE MARKET ---", current_slug)

            # 3) Resolve UP token id
            if up_token is None:
                try:
                    up_token = get_up_token_id(current_slug)
                except Exception as e:
                    print("Gamma market lookup failed:", e)
                    time.sleep(0.5)
                    continue

                if up_token is None:
                    print("Could not resolve UP token_id yet. Retrying...")
                    time.sleep(0.5)
                    continue
                print("UP token_id:", up_token)

            # 4) seconds_left from window_start_ts if available; else approximate
            if window_start_ts is None:
                # fallback: approximate to nearest 15m
                window_start_ts = now_ts - (now_ts % 900)

            end_ts = window_start_ts + 900
            sec_left = end_ts - now_ts
            if sec_left <= 0:
                # wait for next market to appear
                time.sleep(0.2)
                continue

            # 5) Fetch BTC price (Coinbase)
            try:
                btc_cur = btc_usd_coinbase()
            except Exception as e:
                print("BTC price fetch failed:", e)
                time.sleep(0.2)
                continue

            # 6) Set target price
            # Best effort: set at start of window if we catch it; otherwise set on first tick we see.
            if btc_target is None:
                btc_target = btc_cur

            # 7) Fetch order book for bid/ask
            try:
                book = clob_book(up_token)
                bid, ask = best_bid_ask(book)
            except Exception as e:
                print("Polymarket book fetch failed:", e)
                time.sleep(0.2)
                continue

            # 8) Theo
            theo = theo_up(grid, sec_left, btc_cur, btc_target)

            # 9) Log one row
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    now_ts, window_start_ts, sec_left, current_slug,
                    btc_cur, btc_target,
                    bid, ask,
                    theo
                ])

            # 10) Print status
            print(f"{time.strftime('%H:%M:%S')} L={sec_left:3d} bid={bid:.4f} ask={ask:.4f} theo={theo:.4f} BTC={btc_cur:,.2f} tgt={btc_target:,.2f}")

            time.sleep(DT)

    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Log saved to:", LOG_PATH)
    finally:
        grid.close()

if __name__ == "__main__":
    main()#!/usr/bin/env python3
import time, json, csv, os
import urllib.request, urllib.parse
import mmap, struct

CLOB_HOST  = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
SLUG_PREFIX = "btc-updown-15m-"

CDF_GRID_PATH = "out/cdf_grid.f32"
CDF_META_PATH = "out/cdf_grid_meta.json"

LOG_PATH = "out/paper_live_log.csv"
DT = 1.0
STRICT_EPS_BP = 0.1  # use bp_req - 0.1 to approximate strict >=

def http_get_json(url: str, timeout=6):
    req = urllib.request.Request(url, headers={"User-Agent": "paper-log/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def btc_usd_coinbase():
    j = http_get_json("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=5)
    return float(j["price"])

def gamma_market_by_slug(slug: str):
    q = urllib.parse.urlencode({"slug": slug})
    arr = http_get_json(f"{GAMMA_HOST}/markets?{q}", timeout=8)
    return arr[0] if isinstance(arr, list) and arr else None

def parse_list_field(x):
    if isinstance(x, list): 
        return x
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return v
        except:
            pass
    return None

def get_up_token_id(slug: str):
    m = gamma_market_by_slug(slug)
    if not m:
        return None
    outcomes = parse_list_field(m.get("outcomes"))
    token_ids = parse_list_field(m.get("clobTokenIds"))
    if not outcomes or not token_ids or len(outcomes) != len(token_ids):
        return None
    for o, tid in zip(outcomes, token_ids):
        if isinstance(o, str) and o.strip().lower() == "up":
            return tid
    return None

def clob_book(token_id: str):
    q = urllib.parse.urlencode({"token_id": token_id})
    return http_get_json(f"{CLOB_HOST}/book?{q}", timeout=6)

def best_bid_ask(book_json):
    bids = book_json.get("bids", [])
    asks = book_json.get("asks", [])
    bid = float(bids[0]["price"]) if bids else float("nan")
    ask = float(asks[0]["price"]) if asks else float("nan")
    return bid, ask

class CDFGrid:
    def __init__(self, grid_path, meta_path):
        with open(meta_path, "r") as f:
            m = json.load(f)
        self.bp10_min = int(m["bp10_min"])
        self.bp10_max = int(m["bp10_max"])
        self.rows, self.cols = map(int, m["shape"])
        self.f = open(grid_path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        try: self.mm.close()
        except: pass
        try: self.f.close()
        except: pass

    def cdf(self, lag, bp):
        if lag < 1: lag = 1
        if lag >= self.rows: lag = self.rows - 1
        bp10 = int(round(bp * 10.0))
        if bp10 < self.bp10_min: return 0.0
        if bp10 > self.bp10_max: return 1.0
        col = bp10 - self.bp10_min
        off = (lag * self.cols + col) * 4
        return struct.unpack_from("<f", self.mm, off)[0]

def theo_up(grid, lag_left, s_cur, s_target):
    if s_cur <= 0 or s_target <= 0:
        return float("nan")
    bp_req = 10000.0 * (s_target / s_cur - 1.0)
    c = grid.cdf(lag_left, bp_req - STRICT_EPS_BP)
    t = 1.0 - c
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    return t

def floor_15m(t):
    return int(t) - (int(t) % 900)

def ensure_log_header():
    os.makedirs("out", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts","window_start","seconds_left","slug",
                "btc_cur","btc_target",
                "up_bid","up_ask",
                "theo_up"
            ])

def main():
    ensure_log_header()

    if not (os.path.exists(CDF_GRID_PATH) and os.path.exists(CDF_META_PATH)):
        print("Missing grid files. Expected:")
        print(" ", CDF_GRID_PATH)
        print(" ", CDF_META_PATH)
        return

    grid = CDFGrid(CDF_GRID_PATH, CDF_META_PATH)

    window_start = floor_15m(time.time())
    slug = f"{SLUG_PREFIX}{window_start}"
    up_token = None
    btc_target = None

    try:
        while True:
            now = time.time()

            new_start = floor_15m(now)
            if new_start != window_start:
                window_start = new_start
                slug = f"{SLUG_PREFIX}{window_start}"
                up_token = None
                btc_target = None
                print("\n--- NEW WINDOW ---", slug)

            end = window_start + 900
            sec_left = int(end - now)
            if sec_left <= 0:
                time.sleep(0.05)
                continue

            if up_token is None:
                up_token = get_up_token_id(slug)
                if up_token is None:
                    print("Waiting for market in Gamma:", slug)
                    time.sleep(0.5)
                    continue

            btc = btc_usd_coinbase()
            if btc_target is None:
                btc_target = btc

            book = clob_book(up_token)
            bid, ask = best_bid_ask(book)

            t = theo_up(grid, sec_left, btc, btc_target)

            ts = int(now)
            with open(LOG_PATH, "a", newline="") as f:
                csv.writer(f).writerow([ts, window_start, sec_left, slug, btc, btc_target, bid, ask, t])

            print(f"{time.strftime('%H:%M:%S')} L={sec_left:3d} bid={bid:.4f} ask={ask:.4f} theo={t:.4f} BTC={btc:,.2f} tgt={btc_target:,.2f}")
            time.sleep(DT)

    finally:
        grid.close()

if __name__ == "__main__":
    main()
