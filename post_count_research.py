#!/usr/bin/env python3
"""Point-in-time backtest of POST_COUNT_NO on closed Polymarket post-count events.

  collect   closed negRisk events of every configured series + hourly CLOB prices
            (+ XTracker post times for a model series)
  backtest  first-hit replay of the configured rule per series, all-history vs recent window
  export    write the per-event replay rows to post_count_no.history_path (the bot's sizing seed);
            --series NAME rewrites only that series and keeps the other rows as they are

Rules, slippage and windows come from config/strategy.json → post_count_no, so the replay
uses the same parameters and the same top_bucket_signal() as the live scanner. A model series
(rule "model") replays count_model() on the exact XTracker counts of the windows ended by
the replayed hour, as the scanner does live.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

import post_count_strategy as pcs

ROOT = Path(__file__).parent
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
XTRACKER = "https://xtracker.polymarket.com/api"
DEFAULT_DB = ROOT / "data" / "post_count_research.sqlite3"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "nostradamus-post-count-research/1.0"


def load_cfg() -> dict:
    return json.loads((ROOT / "config" / "strategy.json").read_text())["post_count_no"]


def get(url: str, **params):
    for attempt in range(4):
        try:
            response = SESSION.get(url, params=params, timeout=40)
            if response.status_code == 429:
                time.sleep(2 + 2 * attempt)
                continue
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def ts(value) -> int:
    return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())


def json_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value or "[]")
        return result if isinstance(result, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS events(
            event_id TEXT PRIMARY KEY, series TEXT, slug TEXT, title TEXT,
            created_ts INT, start_ts INT, end_ts INT, n_markets INT, volume REAL);
        CREATE TABLE IF NOT EXISTS markets(
            market_id TEXT PRIMARY KEY, event_id TEXT, title TEXT, yes_token TEXT,
            resolved_yes INT, fee_rate REAL, volume REAL);
        CREATE TABLE IF NOT EXISTS prices(token_id TEXT, ts INT, price REAL, PRIMARY KEY(token_id, ts));
        CREATE TABLE IF NOT EXISTS fetched(token_id TEXT PRIMARY KEY, n INT, fetched_ts INT);
        CREATE TABLE IF NOT EXISTS posts(handle TEXT, post_id TEXT, ts INT, PRIMARY KEY(handle, post_id));
    """)
    return conn


def collect(args) -> None:
    cfg = load_cfg()
    conn = connect(args.db)
    for name, series in cfg["series"].items():
        if series.get("model"):
            collect_posts(conn, cfg, name)
            continue  # same events as the price rule of its series_id
        kept = 0
        for offset in range(0, 5000, 100):
            page = get(f"{GAMMA}/events", series_id=series["series_id"], limit=100,
                       offset=offset, closed="true")
            if not page:
                break
            for event in page:
                markets = event.get("markets") or []
                rows = []
                for market in markets:
                    tokens = json_list(market.get("clobTokenIds"))
                    prices = [float(v) for v in json_list(market.get("outcomePrices"))]
                    if not pcs.bucket_bounds(market.get("groupItemTitle")) or len(tokens) < 2 or not prices:
                        continue
                    fee = float((market.get("feeSchedule") or {}).get("rate") or 0) if market.get("feesEnabled") else 0.0
                    rows.append((str(market["id"]), str(event["id"]), str(market.get("groupItemTitle")),
                                 str(tokens[0]), int(prices[0] >= 0.95), fee,
                                 float(market.get("volumeNum") or market.get("volume") or 0)))
                if not event.get("negRisk") or len(rows) < 3 or sum(r[4] for r in rows) != 1:
                    continue
                start = event.get("startTime") or event.get("startDate")
                conn.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?,?,?)", (
                    str(event["id"]), name, event.get("slug"), event.get("title"),
                    ts(event.get("createdAt") or event.get("startDate")), ts(start), ts(event["endDate"]),
                    len(rows), float(event.get("volume") or 0)))
                conn.executemany("INSERT OR REPLACE INTO markets VALUES (?,?,?,?,?,?,?)", rows)
                kept += 1
            if len(page) < 100:
                break
        conn.commit()
        print(f"{name}: {kept} closed events", flush=True)

    todo = conn.execute("""
        SELECT m.yes_token, e.created_ts, e.end_ts FROM markets m JOIN events e USING(event_id)
        WHERE m.yes_token NOT IN (SELECT token_id FROM fetched)
    """).fetchall()
    print(f"price histories to fetch: {len(todo)}", flush=True)

    def fetch(token: str, start: int, end: int) -> list[dict]:
        points, cursor = [], start
        while cursor < end:
            stop = min(end, cursor + 13 * 86400)
            payload = get(f"{CLOB}/prices-history", market=token, startTs=cursor, endTs=stop, fidelity=60)
            points.extend(payload.get("history") or [])
            cursor = stop
        return points

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, tok, start - 3600, end + 3600): tok for tok, start, end in todo}
        for done, future in enumerate(as_completed(futures), 1):
            token = futures[future]
            try:
                points = future.result()
            except Exception as exc:
                print(f"ERR {token[:12]}: {exc}", flush=True)
                continue
            conn.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?)",
                             [(token, int(p["t"]), float(p["p"])) for p in points])
            conn.execute("INSERT OR REPLACE INTO fetched VALUES (?,?,?)", (token, len(points), int(time.time())))
            if done % 200 == 0:
                conn.commit()
                print(f"prices {done}/{len(todo)}", flush=True)
    conn.commit()


def collect_posts(conn: sqlite3.Connection, cfg: dict, name: str) -> None:
    """XTracker post times covering every collected window of the series (one request)."""
    series = cfg["series"][name]
    start = conn.execute("SELECT MIN(end_ts) FROM events WHERE series=?", (series_key(cfg, name),)).fetchone()[0]
    if start is None:
        print(f"{name}: collect the price-rule series first", flush=True)
        return
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    handle = series["xtracker_handle"]
    posts = get(f"{XTRACKER}/users/{handle}/posts",
                startDate=iso(start - float(series["window_days"]) * 86400), endDate=iso(time.time()))["data"]
    conn.executemany("INSERT OR REPLACE INTO posts VALUES (?,?,?)",
                     [(handle, p["id"], ts(p["createdAt"])) for p in posts])
    conn.commit()
    print(f"{name}: {len(posts)} XTracker posts of @{handle}", flush=True)


def window_counts(conn: sqlite3.Connection, cfg: dict, name: str) -> list[tuple[int, int]]:
    """(end_ts, post count) of every collected window of a model series."""
    series = cfg["series"][name]
    times = [row[0] for row in conn.execute(
        "SELECT ts FROM posts WHERE handle=? ORDER BY ts", (series["xtracker_handle"],))]
    window = float(series["window_days"]) * 86400
    return [(end, bisect.bisect_left(times, end) - bisect.bisect_left(times, end - window))
            for (end,) in conn.execute("SELECT end_ts FROM events WHERE series=?", (series_key(cfg, name),))]


def replay(conn: sqlite3.Connection, cfg: dict) -> list[dict]:
    """First hourly hit of the configured rule per closed event."""
    rows = []
    for name, series in cfg["series"].items():
        slip = float(series.get("backtest_slippage_pct", 1.0)) / 100
        counts = window_counts(conn, cfg, name) if series.get("model") else []
        for event_id, slug, created, end in conn.execute(
            "SELECT event_id, slug, created_ts, end_ts FROM events WHERE series=? ORDER BY end_ts",
            (series_key(cfg, name),),
        ).fetchall():
            markets = conn.execute(
                "SELECT market_id, title, yes_token, resolved_yes, fee_rate, volume FROM markets WHERE event_id=?",
                (event_id,),
            ).fetchall()
            history = {}
            for market_id, _title, token, *_ in markets:
                points = conn.execute("SELECT ts, price FROM prices WHERE token_id=? ORDER BY ts", (token,)).fetchall()
                if points:
                    history[market_id] = ([p[0] for p in points], [p[1] for p in points])
            if len(history) < 3:
                continue
            meta = {m[0]: m for m in markets}
            first = max(created, min(h[0][0] for h in history.values()))
            for hour in range((first // 3600 + 1) * 3600, end, 3600):
                buckets = []
                for market_id, (times, prices) in history.items():
                    idx = bisect.bisect_right(times, hour) - 1
                    if idx >= 0:
                        buckets.append({"market_id": market_id, "yes": prices[idx], "volume": meta[market_id][5],
                                        "bounds": pcs.bucket_bounds(meta[market_id][1])})
                model = pcs.count_model(counts, hour, series) if series.get("model") else None
                top = pcs.top_bucket_signal(buckets, series, (end - hour) / 86400, model)
                if not top:
                    continue
                market = meta[top["market_id"]]
                rows.append({
                    "series": name, "event_id": pcs.signal_id(event_id, name, series), "slug": slug,
                    "market_id": top["market_id"],
                    "bucket": market[1], "entry_ts": hour, "end_ts": end,
                    "yes": round(top["yes"], 4), "cost": round(min(0.999, 1 - top["yes"] + slip), 4),
                    "fee_rate": market[4], "won": int(market[3] == 0), "source": "backtest",
                })
                break
    return rows


def series_key(cfg: dict, name: str) -> str:
    """Collected events are stored under the price-rule series that owns the series_id."""
    sid = cfg["series"][name]["series_id"]
    return next(n for n, s in cfg["series"].items() if s["series_id"] == sid and not s.get("model"))


def print_record(name: str, rows: list[dict], cfg: dict, now: float) -> None:
    record = pcs.track_record(rows, cfg, now)
    decision = pcs.stake_decision(record, cfg, cfg["series"][name].get("mode", "shadow"))
    print(f"\n{name} ({cfg['series'][name].get('mode')})")
    for label in ("all", "recent"):
        w = record[label]
        tag = "all history" if label == "all" else f"last {record['recent_days']:.0f}d"
        print(f"  {tag:>12}: n={w['n']:>3} WR={w['wr']:.1%} [{w['wr_lo']:.1%}..{w['wr_hi']:.1%}] "
              f"breakeven={w['breakeven']:.1%} ROI={w['roi']:+.1%} kelly_lo={w['kelly_lo']:+.3f}")
    drift = record["drift_z"]
    print(f"  → mode={decision['mode']} stake={decision['stake_pct']:.2f}% "
          f"drift_z={'—' if drift is None else f'{drift:+.2f}'}"
          f"{' (drifting → min)' if decision['drifting'] else ''} {decision['reason']}")


def backtest(args) -> None:
    cfg = load_cfg()
    rows = replay(connect(args.db), cfg)
    now = args.now or time.time()
    for name in cfg["series"]:
        series_rows = [r for r in rows if r["series"] == name]
        if series_rows:
            print_record(name, series_rows, cfg, now)


def export(args) -> None:
    cfg = load_cfg()
    target = ROOT / cfg["history_path"]
    names = args.series or list(cfg["series"])
    old = json.loads(target.read_text()) if args.series else {"rules": {}, "rows": []}
    rows = [r for r in old["rows"] if r["series"] not in names]
    rows += [r for r in replay(connect(args.db), cfg) if r["series"] in names]
    rows.sort(key=lambda r: (list(cfg["series"]).index(r["series"]), r["end_ts"]))
    rules = {**old["rules"], **{name: rule_params(cfg["series"][name]) for name in names}}
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rules": {name: rules[name] for name in cfg["series"] if name in rules},
        "rows": rows,
    }
    # One event per line keeps recalibration diffs readable.
    head = json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=1, ensure_ascii=False)
    lines = ",\n".join("  " + json.dumps(row, ensure_ascii=False) for row in rows)
    target.write_text(head[:-2] + ',\n "rows": [\n' + lines + "\n ]\n}\n")
    print(f"wrote {len(rows)} rows to {target}")


def rule_params(series: dict) -> dict:
    """Rule parameters a history was generated with (test_post_count_strategy checks them)."""
    return {k: v for k, v in series.items() if k not in ("label", "mode", "max_slippage_pct")}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = root.add_subparsers(dest="cmd", required=True)
    col = sub.add_parser("collect")
    col.add_argument("--workers", type=int, default=8)
    col.set_defaults(func=collect)
    back = sub.add_parser("backtest")
    back.add_argument("--now", type=float, default=0.0, help="unix ts for the recent window (default: now)")
    back.set_defaults(func=backtest)
    exp = sub.add_parser("export")
    exp.add_argument("--series", action="append", help="rewrite only this series (repeatable)")
    exp.set_defaults(func=export)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
    sys.exit(0)
