"""
POST_COUNT_NO: NO on the top-priced bucket of Polymarket "# tweets/posts" events.

Pure rule and track-record math shared by the bot (scanner/main) and
post_count_research.py, so the backtest and live entries use the same code.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

ROOT = Path(__file__).parent
SUM_YES_MIN = 0.90
SUM_YES_MAX = 1.15


def bucket_bounds(title: str) -> tuple[float, float] | None:
    """'<20' / '20-39' / '500+' -> (lo, hi) with ±inf for open ends."""
    text = str(title or "").replace(",", "").strip()
    nums = [int(value) for value in re.findall(r"\d+", text)]
    if not nums:
        return None
    lower = text.lower()
    if text.startswith("<") or "less than" in lower or "fewer" in lower:
        return -math.inf, float(nums[0] - (1 if text.startswith("<") else 0))
    if "+" in text or "or more" in lower or "more than" in lower:
        return float(nums[0]), math.inf
    return float(nums[0]), float(nums[-1])


def count_model(points: list[tuple[float, float]], now_ts: float, rule: dict) -> dict | None:
    """Recent-counts model from windows ended by now_ts.

    points: (end_ts, post count) of ended windows of the series. The forecast centre is
    the mean count of the last lookback_windows windows; the spread is the empirical
    log(count / that mean) of every earlier window (each against its own predecessors).
    """
    k = int(rule["lookback_windows"])
    window = float(rule["window_days"]) * 86400
    known = sorted(p for p in points if p[0] <= now_ts)

    def base_before(ts: float) -> float | None:
        prev = [count for end, count in known if end <= ts][-k:]
        return sum(prev) / k if len(prev) == k else None

    ratios = []
    for end, count in known:
        base = base_before(end - window)
        if base:
            ratios.append(math.log(max(count, 1) / base))
    base = base_before(now_ts)
    if base is None or len(ratios) < int(rule.get("min_model_history", 12)):
        return None
    return {"base": base, "ratios": ratios}


def bucket_probability(bounds: tuple[float, float], model: dict, bandwidth: float) -> float:
    """P(count in bucket): Gaussian kernel in log space over the model's past ratios."""
    lo, hi = bounds

    def cdf(x: float, mu: float) -> float:
        if x <= 0:
            return 0.0
        return 0.5 * (1 + math.erf((math.log(x) - mu) / (bandwidth * math.sqrt(2))))

    total = 0.0
    for ratio in model["ratios"]:
        mu = math.log(model["base"]) + ratio
        upper = 1.0 if hi == math.inf else cdf(hi + 0.5, mu)
        lower = 0.0 if lo == -math.inf else cdf(lo - 0.5, mu)
        total += upper - lower
    return total / len(model["ratios"])


def top_bucket_signal(buckets: list[dict], rule: dict, days_left: float,
                      model: dict | None = None) -> dict | None:
    """First-hit rule on the highest-YES bucket of a complete group.

    buckets: every tradable bucket of the event, each {"yes": 0..1, ...}; a model rule
    also needs "bounds". A price rule fires when the top YES sits in the band; a model
    rule (rule["model"]) when the market prices the top bucket at least min_edge_pct
    above count_model's probability. Returns the top bucket dict (with "p_model" for a
    model rule), or None when the rule does not fire.
    """
    if len(buckets) < 3:
        return None
    total = sum(float(b["yes"]) for b in buckets)
    # Stale or unseeded books (all buckets at 0.5 right after creation) break the ranking.
    if not SUM_YES_MIN <= total <= SUM_YES_MAX:
        return None
    if not float(rule["min_days"]) <= days_left <= float(rule["max_days"]):
        return None
    top = max(buckets, key=lambda b: (float(b["yes"]), float(b.get("volume") or 0)))
    if rule.get("model"):
        if not model:
            return None
        p_model = bucket_probability(top["bounds"], model, float(rule["bandwidth"]))
        if float(top["yes"]) - p_model < float(rule["min_edge_pct"]) / 100:
            return None
        return {**top, "p_model": p_model}
    yes_pct = float(top["yes"]) * 100
    if not float(rule["yes_min_pct"]) < yes_pct <= float(rule["yes_max_pct"]):
        return None
    return top


def signal_id(event_id: str, series_key: str, rule: dict) -> str:
    """post_count_signals row id: a model series shares Gamma events with the price rule
    of the same series_id, so its rows get their own id and both are measured."""
    return f"{event_id}:{series_key}" if rule.get("model") else str(event_id)


def fee_per_share(price: float, fee_rate: float) -> float:
    return max(0.0, float(fee_rate)) * price * (1.0 - price)


def trade_return(row: dict) -> float:
    """Return on $1 of a NO position held to resolution (fees included)."""
    cost = float(row["cost"])
    effective = cost + fee_per_share(cost, float(row.get("fee_rate") or 0))
    return (1.0 / effective if int(row["won"]) else 0.0) - 1.0


def wilson_bounds(wins: int, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def window_stats(rows: list[dict], z: float) -> dict:
    n = len(rows)
    wins = sum(int(r["won"]) for r in rows)
    rets = [trade_return(r) for r in rows]
    costs = [
        float(r["cost"]) + fee_per_share(float(r["cost"]), float(r.get("fee_rate") or 0))
        for r in rows
    ]
    lo, hi = wilson_bounds(wins, n, z)
    breakeven = sum(costs) / n if n else 1.0
    kelly = (lo - breakeven) / (1 - breakeven) if n and breakeven < 1 else 0.0
    return {
        "n": n,
        "wins": wins,
        "wr": wins / n if n else 0.0,
        "wr_lo": lo,
        "wr_hi": hi,
        "breakeven": breakeven,
        "roi": sum(rets) / n if n else 0.0,
        "pnl_units": sum(rets),
        "kelly_lo": kelly,
    }


def two_proportion_z(a: dict, b: dict) -> float | None:
    """Pooled z for WR(a) − WR(b); None when a sample is empty or WR is 0/1 overall."""
    if not a["n"] or not b["n"]:
        return None
    pooled = (a["wins"] + b["wins"]) / (a["n"] + b["n"])
    if pooled <= 0 or pooled >= 1:
        return None
    se = math.sqrt(pooled * (1 - pooled) * (1 / a["n"] + 1 / b["n"]))
    return (a["wr"] - b["wr"]) / se


def track_record(rows: list[dict], cfg: dict, now: float | None = None) -> dict:
    """All-history, recent-window and earlier stats for one series (rows deduped per event)."""
    now = time.time() if now is None else now
    stake_cfg = cfg.get("stake", {})
    z = float(stake_cfg.get("wilson_z", 1.645))
    recent_days = float(stake_cfg.get("recent_days", 120))
    cutoff = now - recent_days * 86400
    recent = [r for r in rows if float(r["end_ts"]) >= cutoff]
    earlier = [r for r in rows if float(r["end_ts"]) < cutoff]
    record = {
        "all": window_stats(rows, z),
        "recent": window_stats(recent, z),
        "earlier": window_stats(earlier, z),
        "recent_days": recent_days,
    }
    record["drift_z"] = two_proportion_z(record["recent"], record["earlier"])
    return record


def stake_decision(record: dict, cfg: dict, series_mode: str) -> dict:
    """Stake % of balance sized on all history; the recent window is a drift guard.

    stake = clamp(kelly_fraction * Kelly(Wilson lower WR, avg breakeven), min_pct, max_pct)
    on the whole track record: a 120-day window of ~30 events has a Wilson interval too
    wide to ever clear a ~84% breakeven. Once the recent window has min_recent_samples
    events and its WR is significantly below the earlier history (z ≤ −drift_z), the
    stake drops to min_pct. Kill switch to shadow when a window is significantly losing.
    """
    stake_cfg = cfg.get("stake", {})
    kill_cfg = cfg.get("kill_switch", {})
    min_pct = float(stake_cfg.get("min_pct", 1.0))
    max_pct = float(stake_cfg.get("max_pct", 5.0))
    fraction = float(stake_cfg.get("kelly_fraction", 0.5))
    min_recent = int(stake_cfg.get("min_recent_samples", 15))
    drift_limit = float(stake_cfg.get("drift_z", 1.645))
    min_live = int(stake_cfg.get("min_live_samples", 20))
    kelly = max(0.0, record["all"]["kelly_lo"])
    stake_pct = min(max_pct, max(min_pct, fraction * kelly * 100))
    drift_z = record.get("drift_z")
    drifting = (
        record["recent"]["n"] >= min_recent
        and drift_z is not None
        and drift_z <= -drift_limit
    )
    if drifting:
        stake_pct = min_pct

    mode = series_mode
    reason = ""
    kill_min = int(kill_cfg.get("min_samples", 20))
    if mode == "live" and record["all"]["n"] < min_live:
        # No track record is not a reason to trade at the floor stake.
        mode = "shadow"
        reason = f"мало истории: n={record['all']['n']} < {min_live}"
    if mode == "live":
        for name, window in (("all", record["all"]), ("recent", record["recent"])):
            # Significantly below breakeven: the upper WR bound cannot pay the average price.
            if window["n"] >= kill_min and window["wr_hi"] < window["breakeven"]:
                mode = "shadow"
                reason = f"kill switch: {name} WR≤{window['wr_hi']:.0%} < breakeven {window['breakeven']:.0%}"
                break
    return {
        "mode": mode,
        "stake_pct": stake_pct if mode == "live" else 0.0,
        "drifting": drifting,
        "reason": reason,
    }


def promotion_status(forward: dict, record: dict, cfg: dict) -> dict:
    """Shadow → live readiness: enough forward shadow results with the backtest dynamics."""
    prom = cfg.get("promotion", {})
    need = int(prom.get("min_forward_samples", 20))
    ready = (
        forward["n"] >= need
        and forward["roi"] > 0
        and record["all"]["kelly_lo"] > 0
    )
    return {"need": need, "have": forward["n"], "ready": ready}


def load_history(path: str | Path) -> list[dict]:
    """Backtest seed rows exported by post_count_research.py export.

    A configured but missing file raises: callers fail closed instead of trading
    with an empty track record.
    """
    if not path:
        return []
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    payload = json.loads(target.read_text())
    return list(payload.get("rows", []))


def merge_rows(seed: list[dict], forward: list[dict]) -> list[dict]:
    """One row per event: forward (live/shadow) results replace backtest rows."""
    merged = {str(r["event_id"]): r for r in seed}
    merged.update({str(r["event_id"]): r for r in forward})
    return sorted(merged.values(), key=lambda r: float(r["end_ts"]))


def series_states(cfg: dict, forward_rows: list[dict], now: float | None = None) -> dict:
    """Per-series track record, stake decision and shadow→live readiness.

    forward_rows: resolved post_count_signals rows (live fills and shadow quotes).
    """
    now = time.time() if now is None else now
    z = float(cfg.get("stake", {}).get("wilson_z", 1.645))
    seed = load_history(cfg.get("history_path", ""))
    states = {}
    for name, series in (cfg.get("series") or {}).items():
        own_seed = [r for r in seed if r.get("series") == name]
        own_forward = [r for r in forward_rows if r.get("series") == name]
        record = track_record(merge_rows(own_seed, own_forward), cfg, now)
        config_mode = str(series.get("mode", "shadow"))
        decision = stake_decision(record, cfg, config_mode)
        shadow_forward = window_stats([r for r in own_forward if r.get("mode") == "shadow"], z)
        states[name] = {
            "label": series.get("label", name),
            "config_mode": config_mode,
            "record": record,
            "decision": decision,
            "forward_live": window_stats([r for r in own_forward if r.get("mode") == "live"], z),
            "forward_shadow": shadow_forward,
            "promotion": promotion_status(shadow_forward, record, cfg),
            "seed_n": len(own_seed),
        }
    return states
