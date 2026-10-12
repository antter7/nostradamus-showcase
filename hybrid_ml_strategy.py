"""
HYBRID_ML_EDGE strategy.

Uses Predict Collector as the first-stage data source:
- DeepSeek probability from collector DB (`predictions`, or the Flash column of
  `model_comparisons` with probability_source="flash")
- hard filters from the hybrid backtest
- LightGBM raw score trained on resolved collector predictions

The raw score is a ranking score, not a calibrated probability.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
import warnings
from html import escape
from pathlib import Path
from typing import Optional

from category_classifier import classify_market_theme
STRATEGY_HYBRID_ML_EDGE = "HYBRID_ML_EDGE"

log = logging.getLogger("hybrid_ml_strategy")


# Collector rows with unusable AI output; legacy `predictions` source only.
_VALID_AI_REASONING_SQL = """
                  AND lower(COALESCE(ai_reasoning, '')) NOT LIKE 'parse error%'
                  AND lower(COALESCE(ai_reasoning, '')) NOT LIKE 'error:%'
                  AND lower(COALESCE(ai_reasoning, '')) <> 'no api key'"""

# One row per market: its first Flash forecast. Before 2026-08-31 the collector
# re-asked a market until the edge looked big enough; later attempts are
# selected on noise, so only the first one counts.
_FLASH_ROWS_SQL = """
    SELECT mc.market_id, mc.question, mc.category, mc.volume, mc.market_price,
           mc.flash_probability AS ai_probability, mc.days_to_resolve, mc.end_date,
           mc.predicted_at, mc.resolved, mc.resolved_yes, mc.resolved_at
    FROM model_comparisons mc
    JOIN (
        SELECT market_id, MIN(predicted_at) AS first_at
        FROM model_comparisons
        GROUP BY market_id
    ) first ON first.market_id = mc.market_id AND first.first_at = mc.predicted_at
    WHERE mc.flash_probability IS NOT NULL
      AND mc.market_price > 0
      AND mc.market_price < 1
      AND mc.predicted_at > 0
"""

_CRYPTO_RE = re.compile(
    r"\b(bitcoin|btc|ethereum|eth|solana|xrp|ripple|dogecoin|doge|bnb|cardano|"
    r"hyperliquid|litecoin|ltc|avalanche|avax|chainlink|pepe|shiba)\b"
)
# Ordinary words too ("sol", "hype", "sui"); crypto only inside the Crypto category.
_CRYPTO_AMBIGUOUS_RE = re.compile(r"\b(sol|hype|sui)\b")
_PRICE_WORDS_RE = re.compile(
    r"\b(price|above|below|between|reach|dip|hit|close|trade|up or down|greater|less)\b|\$"
)
_EQUITY_RE = re.compile(
    r"\b(s&p|spx|spy|qqq|nasdaq|dow jones|stock|stocks|shares|ndx|russell)\b|"
    r"\([a-z]{1,5}\)\s+(?:hit|close|up or down|finish|open)"
)
_COMMODITY_RE = re.compile(r"\b(gold|silver|crude|oil|wti|brent|natural gas|copper)\b")
_MACRO_RE = re.compile(
    r"\b(cpi|inflation|fed|fomc|interest rates?|rate cut|rate hike|bps|gdp|unemployment|"
    r"jobs report|nonfarm|payrolls|pce|ecb|boe|boj|treasury yield|key rate|selic|central bank)\b"
)
_ELECTION_RE = re.compile(
    r"\b(election|elections|vote|votes|voting|primary|mayor|mayoral|governor|senate|"
    r"parliament|parliamentary|president|presidential|by-election|referendum|"
    r"prime minister|chancellor|seats|party|coalition|poll|nominee|nomination for)\b"
)
_SOCIAL_RE = re.compile(r"\b(tweet|tweets|posts|truth social|retweets|followers|views|subscribers)\b")
_BOX_OFFICE_RE = re.compile(r"\b(box office|opening weekend|gross)\b")
_AWARDS_RE = re.compile(
    r"\b(award|awards|oscar|oscars|emmy|emmys|grammy|grammys|vma|vmas|golden globe|"
    r"nominated|nomination|best picture|best actor|best actress)\b"
)
_WEATHER_RE = re.compile(r"\b(temperature|°f|°c|rain|snow|hurricane|storm|weather)\b")
_GEOPOLITICS_RE = re.compile(
    r"\b(military|strike|strikes|ceasefire|invade|invasion|war|missile|troops|sanctions|"
    r"tariff|tariffs|nuclear|hostage|hostages|attack|target|targets|drone|drones)\b"
)
_RELEASE_RE = re.compile(r"\b(release|released|launch|launches|announce|announced|ipo)\b")

# Applied after dates are removed, so day spans like "August 17-23" are not brackets.
_RANGE_RE = re.compile(
    r"\bbetween\b.+\b(?:and|-|–)\b|[$€£]?\d[\d,.]*\s*(?:-|–|to)\s*[$€£]?\d|"
    r"\bor (?:more|fewer|less|higher|lower)\b|\bat (?:least|most)\b|\d\+|[<>≤≥]\s*[$€£]?\d"
)
_THRESHOLD_RE = re.compile(
    r"\b(above|below|over|under|greater than|less than|more than|fewer than|reach|dip to|"
    r"hit|exceed|at or above|at or below|up or down)\b"
)
_DEADLINE_RE = re.compile(
    r"\b(by|before|until)\s+(?:the end of\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"q[1-4]|\d{4}|end of|monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today)"
)
_MONTHS = (
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS})\b\.?(?:\s+\d{{1,2}}(?:st|nd|rd|th)?(?:\s*[-–]\s*\d{{1,2}}(?:st|nd|rd|th)?\b)?)?"
    rf"(?:,?\s+\d{{4}})?"
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER_RE = re.compile(
    r"[$€£]?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|°f|°c)|\s*(?:billion|million|bn|k|m|b)\b)?"
)
# Bump whenever question_family, question_shape, normalize_question or the frame
# columns change: a saved model is only reused with the code it was trained with.
FEATURE_VERSION = 2


def question_family(question: str, category: str = "") -> str:
    """Coarse market family from the title: what kind of event is being priced."""
    text = str(question or "").lower()
    category = str(category or "")
    if category in {"Sports", "Esports", "Weather"}:
        return category.lower()
    crypto = _CRYPTO_RE.search(text) or (category == "Crypto" and _CRYPTO_AMBIGUOUS_RE.search(text))
    if crypto and _PRICE_WORDS_RE.search(text):
        return "crypto_price"
    if crypto:
        return "crypto_other"
    if _EQUITY_RE.search(text):
        return "equity"
    if _COMMODITY_RE.search(text) and _PRICE_WORDS_RE.search(text):
        return "commodity"
    if _MACRO_RE.search(text):
        return "macro"
    if _SOCIAL_RE.search(text):
        return "social_count"
    if _BOX_OFFICE_RE.search(text):
        return "box_office"
    if _AWARDS_RE.search(text):
        return "awards"
    if _ELECTION_RE.search(text):
        return "election"
    if _WEATHER_RE.search(text):
        return "weather"
    if _GEOPOLITICS_RE.search(text):
        return "geopolitics"
    if _RELEASE_RE.search(text):
        return "release"
    return "other"


def question_shape(question: str) -> str:
    """Structure of the proposition: range bracket, threshold, deadline or plain."""
    text = str(question or "").lower()
    undated = _DATE_RE.sub(" ", _ISO_DATE_RE.sub(" ", text))
    if _RANGE_RE.search(undated):
        return "range"
    if _THRESHOLD_RE.search(text):
        return "threshold"
    if _DEADLINE_RE.search(text):
        return "deadline"
    return "plain"


def normalize_question(question: str) -> str:
    """Title for the text model: dates and numbers become placeholders.

    Raw TF-IDF memorised strikes and days ("above 68", "august 17") of a few
    market ladders; placeholders keep the wording without those tokens.
    """
    text = str(question or "").lower()
    text = _ISO_DATE_RE.sub(" datetoken ", text)
    text = _DATE_RE.sub(" datetoken ", text)
    text = _NUMBER_RE.sub(" numtoken ", text)
    return re.sub(r"\s+", " ", text).strip()


DRIFT_NUMERIC = ["ai_probability", "market_price", "edge_abs", "entry_cost", "days_to_resolve", "log_volume"]
DRIFT_CATEGORICAL = ["category", "family", "side"]
DRIFT_PSI_WARN = 0.25          # population stability index: >0.25 is a large shift
DRIFT_MIN_NEW_ROWS = 20


def _psi_numeric(expected, actual, bins: int = 10) -> float:
    import numpy as np

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) < bins or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _psi_categorical(expected, actual) -> float:
    import numpy as np
    import pandas as pd

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    e = pd.Series(expected).value_counts(normalize=True)
    a = pd.Series(actual).value_counts(normalize=True)
    keys = e.index.union(a.index)
    e = np.clip(e.reindex(keys, fill_value=0).to_numpy(), 1e-4, None)
    a = np.clip(a.reindex(keys, fill_value=0).to_numpy(), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _auc(y, score) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, score)) if len(set(y)) > 1 else None
    except Exception:
        return None


def _logit(p: float) -> float:
    p = min(max(float(p), 0.01), 0.99)
    return math.log(p / (1.0 - p))


FEATURE_COLS = [
    "question",
    "category",
    "side",
    "price_band",
    "cost_band",
    "edge_band",
    "days_band",
    "volume_band",
    "market_price",
    "ai_probability",
    "edge_signed",
    "edge_abs",
    "entry_cost",
    "days_to_resolve",
    "log_volume",
]

LEGACY_CATEGORICAL = ["category", "side", "price_band", "cost_band", "edge_band", "days_band", "volume_band"]
LEGACY_NUMERIC = [
    "market_price",
    "ai_probability",
    "edge_signed",
    "edge_abs",
    "entry_cost",
    "days_to_resolve",
    "log_volume",
]
EXTRA_CATEGORICAL = ["family", "shape"]
# Artifacts saved before feature_spec existed were trained like this.
LEGACY_FEATURE_SPEC = {
    "probability_source": "predictions",
    "text_features": True,
    "text_normalize": False,
    "extra_features": False,
    "volume_features": True,
    "train_filters": {},
    "feature_version": 1,
}
EXTRA_NUMERIC = ["logit_price", "logit_ai", "logit_diff"]


def _price_band(price: float) -> str:
    if price < 0.05:
        return "00_05"
    if price < 0.10:
        return "05_10"
    if price < 0.20:
        return "10_20"
    if price < 0.40:
        return "20_40"
    if price < 0.60:
        return "40_60"
    if price < 0.80:
        return "60_80"
    if price < 0.95:
        return "80_95"
    return "95_100"


def _edge_band(edge: float) -> str:
    edge = abs(edge)
    if edge < 0.15:
        return "10_15"
    if edge < 0.20:
        return "15_20"
    if edge < 0.25:
        return "20_25"
    if edge < 0.30:
        return "25_30"
    if edge < 0.40:
        return "30_40"
    if edge < 0.60:
        return "40_60"
    return "60_plus"


def _days_band(days: float) -> str:
    if days < 1:
        return "lt_1d"
    if days < 3:
        return "1_3d"
    if days < 7:
        return "3_7d"
    return "7d_plus"


def _volume_band(volume: float) -> str:
    if volume < 10_000:
        return "2_10k"
    if volume < 100_000:
        return "10_100k"
    return "100k_plus"


def _jsonable_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


class HybridMLStrategy:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.collector_db_path = Path(
            self.cfg.get("collector_db_path", "data/predictions.db")
        )
        self.model_path = self._resolve_model_path(self.cfg.get("model_path", "data/hybrid_ml_edge_model.joblib"))
        self._model = None
        self._model_mtime = 0.0
        self._model_fit_at = 0.0
        self._cache_loaded_at = 0.0
        self._model_resolved_count = 0
        self._last_comparison = {}
        self._last_candidate_checked_at = 0.0
        self._last_candidate_resolved_count = 0
        self._active_filters: dict = {}
        self._open_predictions: dict[str, dict] = {}
        self._deps_ok = None
        self._last_drift: dict = {}
        self.probability_source = str(self.cfg.get("probability_source", "predictions") or "predictions")
        if self.probability_source not in {"predictions", "flash"}:
            raise ValueError(f"unknown hybrid_ml_edge.probability_source: {self.probability_source}")

    def feature_spec(self) -> dict:
        """What the model is trained on; an artifact with another spec is discarded."""
        return {
            "probability_source": self.probability_source,
            "text_features": bool(self.cfg.get("text_features", True)),
            "text_normalize": bool(self.cfg.get("text_normalize", False)),
            "extra_features": bool(self.cfg.get("extra_features", False)),
            "volume_features": bool(self.cfg.get("volume_features", True)),
            "train_filters": dict(self.cfg.get("train_filters", {}) or {}),
            "feature_version": FEATURE_VERSION if self.cfg.get("extra_features") or self.cfg.get("text_normalize") else 1,
        }

    def feature_columns(self) -> list[str]:
        spec = self.feature_spec()
        cols = ["question_norm" if spec["text_normalize"] else "question"] if spec["text_features"] else []
        cols += self._categorical_columns() + self._numeric_columns()
        return cols

    def _categorical_columns(self) -> list[str]:
        spec = self.feature_spec()
        cols = [c for c in LEGACY_CATEGORICAL if spec["volume_features"] or c != "volume_band"]
        return cols + (EXTRA_CATEGORICAL if spec["extra_features"] else [])

    def _numeric_columns(self) -> list[str]:
        spec = self.feature_spec()
        cols = [c for c in LEGACY_NUMERIC if spec["volume_features"] or c != "log_volume"]
        return cols + (EXTRA_NUMERIC if spec["extra_features"] else [])

    def can_score(self, prediction: dict) -> bool:
        """Cheap pre-check before scoring: skip rows that no side could trade."""
        yes_cfg = self.filter_config()
        no_cfg = self.no_filter_config()
        excluded = set(yes_cfg["exclude_categories"])
        if no_cfg["enabled"]:
            excluded &= set(no_cfg["exclude_categories"])
        if str(prediction.get("category") or "Other") in excluded:
            return False
        caps = [float(yes_cfg.get("max_forecast_age_hours", 0.0) or 0.0)]
        if no_cfg["enabled"]:
            caps.append(float(no_cfg.get("max_forecast_age_hours", 0.0) or 0.0))
        if min(caps) <= 0:  # a side without an age cap may still trade it
            return True
        predicted_at = _jsonable_float(prediction.get("predicted_at"))
        return predicted_at > 0 and (time.time() - predicted_at) / 3600 <= max(caps)

    def _resolve_model_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(__file__).parent / path
        return path

    def _ensure_deps(self) -> bool:
        if self._deps_ok is not None:
            return self._deps_ok
        try:
            import pandas  # noqa: F401
            import numpy  # noqa: F401
            import lightgbm  # noqa: F401
            import sklearn  # noqa: F401
            self._deps_ok = True
        except Exception as exc:
            self._deps_ok = False
            log.warning(f"HYBRID_ML_EDGE disabled: ML deps unavailable: {exc}")
        return self._deps_ok

    def _collector_mtime(self) -> float:
        try:
            return self.collector_db_path.stat().st_mtime
        except OSError:
            return 0.0

    def _load_predictions(self) -> tuple[list[dict], dict[str, dict]]:
        if not self.collector_db_path.exists():
            log.warning(f"Predict Collector DB not found: {self.collector_db_path}")
            return [], {}

        with sqlite3.connect(str(self.collector_db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if self.probability_source == "flash":
                # Void (50-50) resolutions are stored with resolved_yes NULL.
                resolved = [dict(row) for row in conn.execute(
                    _FLASH_ROWS_SQL
                    + " AND mc.resolved=1 AND mc.resolved_yes IS NOT NULL"
                    + " ORDER BY mc.resolved_at ASC, mc.id ASC"
                )]
                open_rows = [dict(row) for row in conn.execute(_FLASH_ROWS_SQL + " AND mc.resolved=0")]
                resolved = [row for row in resolved if self._train_row_ok(row)]
            else:
                resolved = [dict(row) for row in conn.execute(f"""
                    SELECT *
                    FROM predictions
                    WHERE resolved=1
                      AND resolved_yes IS NOT NULL
                      AND market_price > 0
                      AND market_price < 1
                      AND predicted_at > 0{_VALID_AI_REASONING_SQL}
                    ORDER BY resolved_at ASC, id ASC
                """)]
                open_rows = [dict(row) for row in conn.execute(f"""
                    SELECT *
                    FROM predictions
                    WHERE resolved=0
                      AND market_price > 0
                      AND market_price < 1{_VALID_AI_REASONING_SQL}
                """)]
        return resolved, {str(row["market_id"]): row for row in open_rows}

    def _train_row_ok(self, row: dict) -> bool:
        """Training rows match what the strategy can trade (train_filters)."""
        tf = self.cfg.get("train_filters", {}) or {}
        price = _jsonable_float(row.get("market_price"))
        ai_probability = _jsonable_float(row.get("ai_probability"))
        if str(row.get("category") or "Other") in set(tf.get("exclude_categories", [])):
            return False
        if not float(tf.get("min_market_price", 0.0)) <= price <= float(tf.get("max_market_price", 1.0)):
            return False
        if abs(ai_probability - price) < float(tf.get("min_edge_abs", 0.0)):
            return False
        return _jsonable_float(row.get("volume")) >= float(tf.get("min_volume", 0.0))

    def _load_model_artifact(self) -> bool:
        if self._model is not None or not self.model_path.exists():
            return self._model is not None
        try:
            import joblib

            artifact = joblib.load(self.model_path)
            stored_spec = artifact.get("feature_spec", LEGACY_FEATURE_SPEC)
            if stored_spec != self.feature_spec():
                log.warning(
                    "HYBRID_ML_EDGE artifact spec %s != config %s; retraining",
                    stored_spec, self.feature_spec(),
                )
                return False
            self._model = artifact.get("model")
            self._model_fit_at = float(artifact.get("trained_at", 0) or 0)
            self._model_resolved_count = int(artifact.get("resolved_count", 0) or 0)
            self._model_mtime = self._collector_mtime()
            self._last_comparison = artifact.get("last_comparison", {}) or {}
            self._last_candidate_checked_at = float(self._last_comparison.get("checked_at", 0) or 0)
            self._last_candidate_resolved_count = int(
                artifact.get(
                    "last_candidate_resolved_count",
                    self._last_comparison.get("resolved_count", 0),
                )
                or 0
            )
            self._active_filters = artifact.get("active_filters", {}) or {}
            self._last_drift = artifact.get("last_drift", {}) or {}
            log.info(
                "HYBRID_ML_EDGE model loaded: %s rows, trained_at=%s, filters=%s",
                self._model_resolved_count,
                int(self._model_fit_at or 0),
                self._active_filters or "config",
            )
            return self._model is not None
        except Exception as exc:
            log.warning(f"HYBRID_ML_EDGE model artifact load failed: {exc}")
            self._model = None
            return False

    def _save_model_artifact(self) -> None:
        if self._model is None:
            return
        try:
            import joblib

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.model_path.with_suffix(self.model_path.suffix + ".tmp")
            joblib.dump({
                "model": self._model,
                "trained_at": self._model_fit_at,
                "resolved_count": self._model_resolved_count,
                "last_candidate_resolved_count": self._last_candidate_resolved_count,
                "last_comparison": self._last_comparison,
                "active_filters": self._active_filters,
                "feature_spec": self.feature_spec(),
                "last_drift": self._last_drift,
                "strategy": STRATEGY_HYBRID_ML_EDGE,
            }, tmp_path)
            tmp_path.replace(self.model_path)
        except Exception as exc:
            log.warning(f"HYBRID_ML_EDGE model artifact save failed: {exc}")

    def _make_frame(self, rows: list[dict], *, current_prices: dict[str, float] | None = None):
        import numpy as np
        import pandas as pd

        records = []
        current_prices = current_prices or {}
        for row in rows:
            market_id = str(row.get("market_id", ""))
            market_price = current_prices.get(market_id, _jsonable_float(row.get("market_price")))
            market_price = max(0.001, min(0.999, market_price))
            ai_probability = max(0.0, min(1.0, _jsonable_float(row.get("ai_probability"))))
            volume = max(1.0, _jsonable_float(row.get("volume")))
            days = max(0.0, _jsonable_float(row.get("days_to_resolve")))
            side = "YES" if ai_probability > market_price else "NO"
            entry_cost = market_price if side == "YES" else 1.0 - market_price
            edge = ai_probability - market_price
            question = row.get("question") or ""
            category = row.get("category") or "Other"
            record = {
                "market_id": market_id,
                "question": question,
                "question_norm": normalize_question(question),
                "category": category,
                "family": question_family(question, category),
                "shape": question_shape(question),
                "logit_price": _logit(market_price),
                "logit_ai": _logit(ai_probability),
                "logit_diff": _logit(ai_probability) - _logit(market_price),
                "predicted_at": _jsonable_float(row.get("predicted_at")),
                "resolved_at": _jsonable_float(row.get("resolved_at")),
                "side": side,
                "market_price": market_price,
                "ai_probability": ai_probability,
                "volume": volume,
                "days_to_resolve": days,
                "edge_signed": edge,
                "edge_abs": abs(edge),
                "entry_cost": entry_cost,
                "log_volume": math.log10(volume),
                "price_band": _price_band(market_price),
                "cost_band": _price_band(entry_cost),
                "edge_band": _edge_band(edge),
                "days_band": _days_band(days),
                "volume_band": _volume_band(volume),
            }
            if "resolved_yes" in row and row.get("resolved_yes") is not None:
                resolved_yes = int(row.get("resolved_yes"))
                record["success"] = int((side == "YES" and resolved_yes == 1) or (side == "NO" and resolved_yes == 0))
            records.append(record)
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["question"] = df["question"].fillna("")
        df["question_norm"] = df["question_norm"].fillna("")
        df["category"] = df["category"].fillna("Other")
        df["success"] = df.get("success", 0)
        return df.replace([np.inf, -np.inf], 0)

    def _build_model(self):
        from lightgbm import LGBMClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        spec = self.feature_spec()
        categorical = self._categorical_columns()
        numeric = self._numeric_columns()
        transformers = []
        if spec["text_features"]:
            transformers.append((
                "title_tfidf",
                TfidfVectorizer(
                    max_features=int(self.cfg.get("max_text_features", 1200)),
                    ngram_range=(1, 2),
                    min_df=int(self.cfg.get("min_df", 2)),
                ),
                "question_norm" if spec["text_normalize"] else "question",
            ))
        transformers += [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
        pre = ColumnTransformer(transformers=transformers, sparse_threshold=0.80)
        clf = LGBMClassifier(
            n_estimators=int(self.cfg.get("lgb_n_estimators", 180)),
            learning_rate=float(self.cfg.get("lgb_learning_rate", 0.035)),
            num_leaves=int(self.cfg.get("lgb_num_leaves", 9)),
            min_child_samples=int(self.cfg.get("lgb_min_child_samples", 18)),
            subsample=float(self.cfg.get("lgb_subsample", 0.85)),
            colsample_bytree=float(self.cfg.get("lgb_colsample_bytree", 0.85)),
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    def filter_config(self, overrides: dict | None = None) -> dict:
        result = {
            "score_threshold": float(self.cfg.get("score_threshold", 0.50)),
            "exclude_categories": list(self.cfg.get("exclude_categories", ["Crypto", "Science"])),
            "min_edge_abs": float(self.cfg.get("min_edge_abs", 0.10)),
            "max_edge_abs": float(self.cfg.get("max_edge_abs", 0.40)),
            "min_entry_cost": float(self.cfg.get("min_entry_cost", 0.0)),
            "max_entry_cost": float(self.cfg.get("max_entry_cost", 1.0)),
            "min_days_to_resolve": float(self.cfg.get("min_days_to_resolve", 0.25)),
            "max_days_to_resolve": float(self.cfg.get("max_days_to_resolve", 10.0)),
            "min_volume": float(self.cfg.get("min_volume", 2000)),
            "ai_probability_min": float(self.cfg.get("ai_probability_min", 0.0)),
            "max_forecast_age_hours": float(self.cfg.get("max_forecast_age_hours", 0.0)),
        }
        result.update(self._active_filters or {})
        result.update(overrides or {})
        if not self.cfg.get("edge_range_enabled", True):
            result.update(min_edge_abs=0.0, max_edge_abs=1.0)
        return result

    def no_filter_config(self, overrides: dict | None = None) -> dict:
        cfg = self.cfg.get("no_side", {}) or {}
        result = {
            "enabled": bool(cfg.get("enabled", False)),
            "ai_probability_max": float(cfg.get("ai_probability_max", 0.10)),
            "score_threshold": float(cfg.get("score_threshold", self.cfg.get("score_threshold", 0.50))),
            "exclude_categories": list(cfg.get("exclude_categories", self.cfg.get("exclude_categories", []))),
            "exclude_themes": list(cfg.get("exclude_themes", [])),
            "min_edge_abs": float(cfg.get("min_edge_abs", self.cfg.get("min_edge_abs", 0.10))),
            "max_edge_abs": float(cfg.get("max_edge_abs", self.cfg.get("max_edge_abs", 0.60))),
            "min_entry_cost": float(cfg.get("min_entry_cost", self.cfg.get("min_entry_cost", 0.0))),
            "max_entry_cost": float(cfg.get("max_entry_cost", self.cfg.get("max_entry_cost", 1.0))),
            "min_days_to_resolve": float(cfg.get("min_days_to_resolve", self.cfg.get("min_days_to_resolve", 0.25))),
            "max_days_to_resolve": float(cfg.get("max_days_to_resolve", self.cfg.get("max_days_to_resolve", 10.0))),
            "min_volume": float(cfg.get("min_volume", self.cfg.get("min_volume", 2000))),
            "max_forecast_age_hours": float(
                cfg.get("max_forecast_age_hours", self.cfg.get("max_forecast_age_hours", 0.0))
            ),
        }
        result.update(overrides or {})
        if not cfg.get("edge_range_enabled", self.cfg.get("edge_range_enabled", True)):
            result.update(min_edge_abs=0.0, max_edge_abs=1.0)
        return result

    @staticmethod
    def _passes_common_filters(scored: dict, fc: dict) -> bool:
        if scored["category"] in set(fc["exclude_categories"]):
            return False
        theme = scored.get("theme") or classify_market_theme(str(scored.get("question") or ""))
        if theme in set(fc.get("exclude_themes", [])):
            return False
        if scored["volume"] < float(fc["min_volume"]):
            return False
        if not (float(fc["min_days_to_resolve"]) <= scored["days_to_resolve"] <= float(fc["max_days_to_resolve"])):
            return False
        if not (float(fc["min_edge_abs"]) <= scored["edge_abs"] < float(fc["max_edge_abs"])):
            return False
        if float(scored.get("entry_cost", 0.0)) > float(fc.get("max_entry_cost", 1.0)):
            return False
        if float(scored.get("entry_cost", 0.0)) < float(fc.get("min_entry_cost", 0.0)):
            return False
        max_age = float(fc.get("max_forecast_age_hours", 0.0) or 0.0)
        if max_age > 0 and float(scored.get("forecast_age_hours", 0.0)) > max_age:
            return False
        if scored["score"] < float(fc["score_threshold"]):
            return False
        return True

    def _promotion_filter_candidates(self) -> list[dict]:
        candidates = self.cfg.get("promotion_filter_candidates")
        if candidates is None:
            candidates = [
                {"name": "edge_max_0.60", "max_edge_abs": 0.60},
                {"name": "edge_max_0.80", "max_edge_abs": 0.80},
                {"name": "min_days_0.10", "min_days_to_resolve": 0.10},
                {"name": "edge_max_0.60_min_days_0.10", "max_edge_abs": 0.60, "min_days_to_resolve": 0.10},
            ]

        normalized = [{"name": "model_only", "filters": {}}]
        for idx, item in enumerate(candidates):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or f"candidate_{idx + 1}")
            filters = {k: v for k, v in item.items() if k != "name"}
            normalized.append({"name": name, "filters": filters})
        return normalized

    def _trade_decisions(self, model, eval_df, filter_overrides: dict | None = None):
        import numpy as np

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            scores = model.predict_proba(eval_df[self.feature_columns()])[:, 1]
        pnl = []
        wins = 0
        trades = 0
        for (_, row), score in zip(eval_df.iterrows(), scores):
            rec = row.to_dict()
            rec["score"] = float(score)
            if not self.passes(rec, filter_overrides=filter_overrides):
                pnl.append(0.0)
                continue
            trades += 1
            if int(rec.get("success", 0)):
                wins += 1
                pnl.append(1.0 / max(float(rec["entry_cost"]), 0.001) - 1.0)
            else:
                pnl.append(-1.0)
        pnl_arr = np.asarray(pnl, dtype=float)
        return {
            "pnl_by_market": pnl_arr,
            "pnl": float(pnl_arr.sum()),
            "trades": trades,
            "wins": wins,
            "win_rate": wins / trades if trades else 0.0,
        }

    def _segment_deltas(self, eval_df, diff) -> dict:
        import numpy as np

        segments = []
        for label, col in (
            ("cat", "category"),
            ("price", "price_band"),
            ("edge", "edge_band"),
            ("days", "days_band"),
            ("vol", "volume_band"),
        ):
            if col not in eval_df:
                continue
            values = eval_df[col].fillna("NA").astype(str).to_numpy()
            for value in sorted(set(values)):
                mask = values == value
                changed = int((np.abs(diff[mask]) > 1e-9).sum())
                delta = float(diff[mask].sum())
                if changed == 0 and abs(delta) < 1e-9:
                    continue
                segments.append({
                    "segment": f"{label}:{value}",
                    "n": int(mask.sum()),
                    "changed": changed,
                    "delta_pnl": delta,
                })

        return {
            "better_segments": sorted(
                [item for item in segments if item["delta_pnl"] > 0],
                key=lambda item: item["delta_pnl"],
                reverse=True,
            )[:3],
            "worse_segments": sorted(
                [item for item in segments if item["delta_pnl"] < 0],
                key=lambda item: item["delta_pnl"],
            )[:3],
        }

    def _compare_candidate(
        self,
        old_model,
        candidate_model,
        eval_df: "object",
        *,
        candidate_name: str,
        candidate_filters: dict | None = None,
    ) -> dict:
        import numpy as np

        old = self._trade_decisions(old_model, eval_df)
        new = self._trade_decisions(candidate_model, eval_df, candidate_filters)
        diff = new["pnl_by_market"] - old["pnl_by_market"]
        if len(diff) > 1:
            rng = np.random.default_rng(42)
            runs = int(self.cfg.get("candidate_bootstrap_runs", 1000))
            sample_means = [
                float(rng.choice(diff, size=len(diff), replace=True).mean())
                for _ in range(max(100, runs))
            ]
            ci_low, ci_high = np.percentile(sample_means, [2.5, 97.5])
        else:
            ci_low = ci_high = float(diff.mean()) if len(diff) else 0.0

        delta_pnl = float(diff.sum())
        min_trades = int(self.cfg.get("promotion_min_trades", 10))
        significant_positive = float(ci_low) > 0
        significant_negative = float(ci_high) < 0
        allow_not_significantly_worse = bool(
            self.cfg.get("promotion_allow_not_significantly_worse", True)
        )
        winrate_floor = old["win_rate"] - float(self.cfg.get("promotion_max_winrate_drop", 0.02))
        non_inferior = not significant_negative
        promote = (
            new["trades"] >= min_trades
            and new["win_rate"] >= winrate_floor
            and (
                (allow_not_significantly_worse and non_inferior)
                or (not allow_not_significantly_worse and significant_positive and delta_pnl > 0)
            )
        )
        return {
            "candidate_name": candidate_name,
            "candidate_filters": candidate_filters or {},
            "active_filters": self._active_filters or {},
            "evaluated_rows": int(len(eval_df)),
            "old_trades": int(old["trades"]),
            "old_wins": int(old["wins"]),
            "old_win_rate": float(old["win_rate"]),
            "old_pnl": float(old["pnl"]),
            "new_trades": int(new["trades"]),
            "new_wins": int(new["wins"]),
            "new_win_rate": float(new["win_rate"]),
            "new_pnl": float(new["pnl"]),
            "delta_pnl": delta_pnl,
            "delta_pnl_mean_ci_low": float(ci_low),
            "delta_pnl_mean_ci_high": float(ci_high),
            "significant_positive": bool(significant_positive),
            "significant_negative": bool(significant_negative),
            "non_inferior": bool(non_inferior),
            "promotion_policy": "non_inferior" if allow_not_significantly_worse else "significant_positive",
            **self._segment_deltas(eval_df, diff),
            "promoted": bool(promote),
            "checked_at": int(time.time()),
        }

    def _promotion_status(self, comparison: dict) -> tuple[str, str]:
        min_trades = int(self.cfg.get("promotion_min_trades", 10))
        max_wr_drop = float(self.cfg.get("promotion_max_winrate_drop", 0.02))
        if comparison.get("promoted"):
            if comparison.get("significant_positive"):
                return "✅", "принята: значимо лучше старой"
            return "✅", "принята: не хуже старой"
        if int(comparison.get("new_trades", 0) or 0) < min_trades:
            return "⚠️", f"старая оставлена: мало holdout сделок ({comparison.get('new_trades', 0)}/{min_trades})"
        if float(comparison.get("new_win_rate", 0) or 0) < float(comparison.get("old_win_rate", 0) or 0) - max_wr_drop:
            return "⚠️", f"старая оставлена: WR ниже допуска {max_wr_drop*100:.0f}пп"
        if comparison.get("significant_negative"):
            return "⚠️", "старая оставлена: новая значимо хуже"
        return "≈", "старая оставлена: gate не пройден"

    def _segment_lines(self, title: str, items: list[dict]) -> list[str]:
        if not items:
            return [f"{title}: —"]
        lines = [f"{title}:"]
        for item in items:
            lines.append(
                f"  • {escape(str(item.get('segment', '?')))} "
                f"Δ {float(item.get('delta_pnl', 0) or 0):+.2f} "
                f"({int(item.get('changed', 0) or 0)}/{int(item.get('n', 0) or 0)})"
            )
        return lines

    def _notify_model_comparison(self, comparison: dict, resolved_count: int) -> None:
        if not bool(self.cfg.get("notify_model_updates", True)):
            return
        try:
            import notifier

            icon, status = self._promotion_status(comparison)
            lines = [
                f"{icon} <b>HYBRID ML retrain</b>",
                escape(status),
                (
                    f"rows: {resolved_count} | holdout: {int(comparison.get('evaluated_rows', 0) or 0)} | "
                    f"candidate: {escape(str(comparison.get('candidate_name', 'model_only')))} | "
                    f"policy: {escape(str(comparison.get('promotion_policy', 'unknown')))}"
                ),
                (
                    f"old: n={int(comparison.get('old_trades', 0) or 0)}, "
                    f"WR {float(comparison.get('old_win_rate', 0) or 0):.0%}, "
                    f"PnL {float(comparison.get('old_pnl', 0) or 0):+.2f}"
                ),
                (
                    f"new: n={int(comparison.get('new_trades', 0) or 0)}, "
                    f"WR {float(comparison.get('new_win_rate', 0) or 0):.0%}, "
                    f"PnL {float(comparison.get('new_pnl', 0) or 0):+.2f} | "
                    f"Δ {float(comparison.get('delta_pnl', 0) or 0):+.2f}"
                ),
                (
                    f"CI/market: "
                    f"[{float(comparison.get('delta_pnl_mean_ci_low', 0) or 0):+.4f}, "
                    f"{float(comparison.get('delta_pnl_mean_ci_high', 0) or 0):+.4f}]"
                ),
            ]
            lines.extend(self._segment_lines("Где лучше", comparison.get("better_segments", []) or []))
            lines.extend(self._segment_lines("Где хуже", comparison.get("worse_segments", []) or []))
            notifier.send_text("\n".join(lines))
        except Exception as exc:
            log.warning(f"HYBRID_ML_EDGE model update notification failed: {exc}")

    def refresh_model(self, force: bool = False) -> bool:
        if not self.enabled or not self._ensure_deps():
            return False
        self._load_model_artifact()

        now = time.time()
        mtime = self._collector_mtime()
        refresh_seconds = float(self.cfg.get("model_refresh_minutes", 60)) * 60
        if (
            not force
            and self._model is not None
            and mtime == self._model_mtime
            and now - self._cache_loaded_at < refresh_seconds
        ):
            return True

        resolved, open_predictions = self._load_predictions()
        self._open_predictions = open_predictions
        self._cache_loaded_at = now

        if self._model is not None and not force:
            min_delta = int(self.cfg.get("min_retrain_resolved_delta", 25))
            max_stale_seconds = float(self.cfg.get("max_model_stale_hours", 24)) * 3600
            artifact_needs_save = False
            if (
                self._last_candidate_checked_at > self._model_fit_at
                and self._last_candidate_resolved_count <= 0
            ):
                self._last_candidate_resolved_count = len(resolved)
                artifact_needs_save = True
            retrain_baseline_count = max(
                self._model_resolved_count,
                self._last_candidate_resolved_count,
            )
            new_resolved = max(0, len(resolved) - retrain_baseline_count)
            freshness_ts = max(self._model_fit_at, self._last_candidate_checked_at)
            if new_resolved < min_delta and now - freshness_ts < max_stale_seconds:
                self._model_mtime = mtime
                if artifact_needs_save:
                    self._save_model_artifact()
                log.info(
                    "HYBRID_ML_EDGE model reuse: %s new resolved < %s, open predictions %s",
                    new_resolved,
                    min_delta,
                    len(self._open_predictions),
                )
                return True

        min_train_rows = int(self.cfg.get("min_train_rows", 250))
        if len(resolved) < min_train_rows:
            log.warning(f"HYBRID_ML_EDGE warmup: {len(resolved)}/{min_train_rows} resolved predictions")
            self._model = None
            self._model_mtime = mtime
            return False

        train_df = self._make_frame(resolved)
        if train_df.empty or train_df["success"].nunique() < 2:
            log.warning("HYBRID_ML_EDGE warmup: not enough class diversity")
            self._model = None
            return False

        old_model = self._model
        model = self._build_model()
        fit_df = train_df
        eval_df = None
        gate_enabled = bool(self.cfg.get("promotion_gate_enabled", True))
        if not gate_enabled:
            # Scheduled refit on every valid resolved row; config filters only.
            # No gate, so watch instead: how the outgoing model did on rows that
            # resolved after its fit, and how far the inputs moved.
            self._active_filters = {}
            if old_model is not None and self._model_fit_at > 0:
                seen = train_df[train_df["resolved_at"] <= self._model_fit_at]
                fresh = train_df[train_df["resolved_at"] > self._model_fit_at]
                try:
                    self._last_drift = self.drift_report(seen, fresh, old_model)
                except Exception as exc:  # monitoring must never block a refit
                    log.warning("HYBRID_ML_EDGE drift report failed: %s", exc)
                    self._last_drift = {"checked_at": int(time.time()), "error": type(exc).__name__}
        if old_model is not None and not force and gate_enabled:
            holdout_rows = int(self.cfg.get("candidate_holdout_rows", 150))
            holdout_rows = max(30, min(holdout_rows, max(30, len(train_df) // 3)))
            if len(train_df) - holdout_rows >= min_train_rows:
                fit_df = train_df.iloc[:-holdout_rows].copy()
                eval_df = train_df.iloc[-holdout_rows:].copy()

        try:
            model.fit(fit_df[self.feature_columns()], fit_df["success"].astype(int))
        except Exception as exc:
            log.warning(f"HYBRID_ML_EDGE model fit failed: {exc}")
            self._model = None
            self._model_mtime = mtime
            return False

        if old_model is not None and eval_df is not None and not force:
            comparisons = [
                self._compare_candidate(
                    old_model,
                    model,
                    eval_df,
                    candidate_name=item["name"],
                    candidate_filters=item["filters"],
                )
                for item in self._promotion_filter_candidates()
            ]
            comparison = max(
                comparisons,
                key=lambda item: (item["promoted"], item["delta_pnl"], item["new_pnl"]),
            )
            self._last_comparison = {
                **comparison,
                "resolved_count": len(resolved),
                "all_candidates": comparisons,
            }
            self._notify_model_comparison(comparison, len(resolved))
            if not comparison["promoted"]:
                self._model = old_model
                self._model_mtime = mtime
                self._last_candidate_checked_at = float(comparison["checked_at"])
                self._last_candidate_resolved_count = len(resolved)
                self._save_model_artifact()
                log.info(
                    "HYBRID_ML_EDGE candidate kept out: best=%s old pnl %.2f, new pnl %.2f, "
                    "delta %.2f, CI [%.4f, %.4f], filters=%s",
                    comparison["candidate_name"],
                    comparison["old_pnl"],
                    comparison["new_pnl"],
                    comparison["delta_pnl"],
                    comparison["delta_pnl_mean_ci_low"],
                    comparison["delta_pnl_mean_ci_high"],
                    comparison["candidate_filters"],
                )
                return True

            final_model = self._build_model()
            final_model.fit(train_df[self.feature_columns()], train_df["success"].astype(int))
            model = final_model
            self._active_filters = comparison["candidate_filters"]
            log.info(
                "HYBRID_ML_EDGE candidate promoted: %s old pnl %.2f, new pnl %.2f, "
                "delta %.2f, CI [%.4f, %.4f], filters=%s",
                comparison["candidate_name"],
                comparison["old_pnl"],
                comparison["new_pnl"],
                comparison["delta_pnl"],
                comparison["delta_pnl_mean_ci_low"],
                comparison["delta_pnl_mean_ci_high"],
                self._active_filters,
            )

        self._model = model
        self._model_mtime = mtime
        self._model_fit_at = now
        self._last_candidate_checked_at = now
        self._model_resolved_count = len(resolved)
        self._last_candidate_resolved_count = len(resolved)
        self._save_model_artifact()
        log.info(
            "HYBRID_ML_EDGE model trained (%s): %s rows, success %.1f%%, open predictions %s",
            self.probability_source,
            len(train_df),
            100 * float(train_df["success"].mean()),
            len(self._open_predictions),
        )
        if not gate_enabled:
            self._notify_retrain(len(train_df), float(train_df["success"].mean()))
        return True

    def drift_report(self, seen, fresh, old_model) -> dict:
        """Input drift (PSI) and out-of-sample quality of the outgoing model.

        `seen` = rows the outgoing model was trained on, `fresh` = rows that
        resolved after its fit. AUC of the model's score is compared with the
        AUC of the side price alone: the research found the score barely beats it.
        """
        report: dict = {"checked_at": int(time.time()), "train_rows": int(len(seen)), "new_rows": int(len(fresh))}
        if len(fresh) == 0 or len(seen) == 0:
            report["warnings"] = []
            return report
        psi = {col: _psi_numeric(seen[col], fresh[col]) for col in DRIFT_NUMERIC}
        psi.update({col: _psi_categorical(seen[col], fresh[col]) for col in DRIFT_CATEGORICAL})
        psi = {k: round(v, 3) for k, v in psi.items() if v == v}
        report["psi"] = psi
        report["success_train"] = round(float(seen["success"].mean()), 3)
        report["success_new"] = round(float(fresh["success"].mean()), 3)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                scores = old_model.predict_proba(fresh[self.feature_columns()])[:, 1]
            y = fresh["success"].astype(int).tolist()
            report["auc_model"] = _auc(y, scores)
            report["auc_price"] = _auc(y, fresh["entry_cost"].tolist())
            threshold = float(self.cfg.get("score_threshold", 0.5))
            kept = fresh[scores >= threshold]
            report["kept_rows"] = int(len(kept))
            report["kept_success"] = round(float(kept["success"].mean()), 3) if len(kept) else None
        except Exception as exc:
            report["model_error"] = type(exc).__name__
        alerts = []
        if len(fresh) < DRIFT_MIN_NEW_ROWS:
            alerts.append(f"мало новых строк ({len(fresh)})")
        for col, value in psi.items():
            if value > DRIFT_PSI_WARN:
                alerts.append(f"сдвиг {col}: PSI {value:.2f}")
        if report.get("auc_model") is not None and report.get("auc_price") is not None:
            if report["auc_model"] < report["auc_price"] - 0.05:
                alerts.append(f"модель хуже цены: AUC {report['auc_model']:.2f} < {report['auc_price']:.2f}")
        if abs(report["success_new"] - report["success_train"]) > 0.10:
            alerts.append(
                f"доля выигрышей AI-стороны {report['success_train']:.0%} → {report['success_new']:.0%}"
            )
        report["warnings"] = alerts
        log.info("HYBRID_ML_EDGE drift: %s", report)
        return report

    @staticmethod
    def drift_lines(report: dict) -> list[str]:
        if report and report.get("error"):
            return [f"drift: отчёт не построен ({report['error']})"]
        if not report or not report.get("new_rows"):
            return ["drift: нет новых строк после прошлого обучения"]
        auc_m, auc_p = report.get("auc_model"), report.get("auc_price")
        psi = report.get("psi") or {}
        worst = max(psi.items(), key=lambda kv: kv[1]) if psi else None
        lines = [
            (
                f"новых строк: {report['new_rows']} | AI-сторона выигрывала "
                f"{report.get('success_train', 0):.0%} → {report.get('success_new', 0):.0%}"
            ),
            (
                "AUC старой модели / цены: "
                + (f"{auc_m:.2f} / {auc_p:.2f}" if auc_m is not None and auc_p is not None else "—")
                + (f" | score≥порога: {report['kept_rows']} шт., выигрыш {report['kept_success']:.0%}"
                   if report.get("kept_rows") else "")
            ),
            f"PSI max: {worst[0]} {worst[1]:.2f}" if worst else "PSI: —",
        ]
        warnings_ = report.get("warnings") or []
        lines.append("⚠️ " + "; ".join(warnings_) if warnings_ else "дрейф в норме")
        return lines

    def _notify_retrain(self, rows: int, success_rate: float) -> None:
        if not bool(self.cfg.get("notify_model_updates", True)):
            return
        try:
            import notifier

            lines = [
                f"🔄 <b>HYBRID ML retrain</b> ({escape(self.probability_source)})",
                f"rows: {rows} | AI-side wins: {success_rate:.0%} | open forecasts: {len(self._open_predictions)}",
                *(escape(line) for line in self.drift_lines(self._last_drift)),
            ]
            notifier.send_text("\n".join(lines))
        except Exception as exc:
            log.warning(f"HYBRID_ML_EDGE model update notification failed: {exc}")

    def prediction_for(self, market_id: str) -> Optional[dict]:
        if not self.refresh_model():
            return None
        return self._open_predictions.get(str(market_id))

    def score(self, prediction: dict, current_yes_price: float, current_days: float, current_volume: float) -> Optional[dict]:
        if self._model is None:
            return None
        row = dict(prediction)
        row["market_price"] = current_yes_price
        row["days_to_resolve"] = current_days
        row["volume"] = current_volume
        frame = self._make_frame([row], current_prices={str(row.get("market_id", "")): current_yes_price})
        if frame.empty:
            return None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            score = float(self._model.predict_proba(frame[self.feature_columns()])[:, 1][0])
        rec = frame.iloc[0].to_dict()
        rec["score"] = score
        predicted_at = _jsonable_float(prediction.get("predicted_at"))
        rec["forecast_age_hours"] = (time.time() - predicted_at) / 3600 if predicted_at > 0 else float("inf")
        return rec

    def passes(self, scored: dict, filter_overrides: dict | None = None) -> bool:
        if not scored:
            return False
        side = str(scored.get("side") or "").upper()
        if side == "YES":
            fc = self.filter_config(filter_overrides)
            if scored["ai_probability"] < float(fc.get("ai_probability_min", 0.0)):
                return False
            return self._passes_common_filters(scored, fc)
        if side == "NO":
            fc = self.no_filter_config(filter_overrides)
            if not fc["enabled"] or scored["ai_probability"] > float(fc["ai_probability_max"]):
                return False
            return self._passes_common_filters(scored, fc)
        return False


def strategy_key() -> str:
    return STRATEGY_HYBRID_ML_EDGE.lower()
