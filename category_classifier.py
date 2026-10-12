"""Shared market category classifier."""

from __future__ import annotations

import re


_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Weather", ["weather", "temperature", "rain", "rainfall", "snow", "wind",
                 "hurricane", "storm", "typhoon", "forecast", "degrees",
                 "celsius", "fahrenheit", "highest temperature",
                 "lowest temperature", "°c", "°f"]),
    ("Crypto", ["bitcoin", "btc", "ethereum", "eth", "crypto", "defi", "nft",
                "token", "blockchain", "solana", "xrp", "doge", "altcoin",
                "hyperliquid"]),
    ("Finance", ["stock", "nasdaq", "s&p", "dow jones", "market cap", "fed",
                 "fomc", "powell", "interest rate", "rate cut", "rate hike",
                 "gdp", "inflation", "cpi", "ppi", "bond", "ipo", "earnings",
                 "bank of canada", "crude oil", "wti", "natural gas", "ng ",
                 "eggs cost", "recession"]),
    ("Politics", ["trump", "biden", "congress", "senate", "election", "president",
                  "government", "minister", "parliament", "vote", "referendum",
                  "democrat", "republican", "policy", "diplomatic meeting",
                  "iran", "gaza", "israel", "russia", "ukraine", "warship",
                  "ceasefire", "capture", "forces enter"]),
    ("Esports", ["esports", "counter-strike", "cs2", "valorant", "dota",
                 "league of legends", "lol:", " bo1", " bo3", " bo5", "iem ",
                 "lck ", "lcs ", "vcl ", "map handicap"]),
    ("Sports", ["sports", "nfl", "nba", "nhl", "mlb", "soccer", "football",
                "basketball", "baseball", "hockey", "tennis", "cricket",
                "grand prix", "f1", "fantasy", "championship",
                "tournament", "league", "playoffs", "superbowl", "world cup",
                "olympic", "french open", "roland garros", "ufc", "golf",
                "spread:", "handicap", "o/u", "total sets"]),
    ("Science", ["ai", "chatgpt", "openai", "nasa", "space", "climate",
                 "earthquake", "spacex", "rocket", "fda", "vaccine"]),
    ("Entertainment", ["oscar", "grammy", "award", "movie", "music", "album",
                       "celebrity", "film", "box office", "netflix", "show",
                       "youtube", "mrbeast", "mandalorian"]),
    ("Media", ["tweet", "tweets", "post", "posts", "followers", "views",
               "subscribers"]),
]


def _has_keyword(text: str, keyword: str) -> bool:
    escaped = re.escape(keyword.lower())
    left = r"(?<![a-z0-9])" if keyword[0].isalnum() else ""
    right = r"(?![a-z0-9])" if keyword[-1].isalnum() else ""
    return bool(re.search(rf"{left}{escaped}{right}", text))


def _classify_text(text: str) -> str | None:
    text = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(_has_keyword(text, kw) for kw in keywords):
            return category
    return None


def classify_category(question: str, tags: list[str]) -> str:
    """Classify by question first, then use tags only as a fallback."""
    lowered = question.lower()
    if _is_clear_sports_result(lowered):
        return "Sports"

    category = _classify_text(question)
    if category:
        return category

    if _looks_like_matchup(lowered):
        return "Esports" if _classify_text(lowered) == "Esports" else "Sports"

    category = _classify_text(" ".join(tags))
    if category:
        return category

    return "Other"


def classify_market_theme(question: str, tags: list[str] | None = None) -> str:
    text = " ".join([question, *(tags or [])]).lower()
    if any(word in text for word in (
        "esports", "counter-strike", "cs2", "valorant", "dota",
        "league of legends", "lol:", " bo1", " bo3", " bo5", "iem ",
        "lck ", "lcs ", "vcl ", "map handicap", "game handicap",
    )):
        return "esports"
    if any(word in text for word in (
        "soccer", "fifa", "fifwc", "world cup", "premier league", "champions league",
        "la liga", "bundesliga", "serie a", "ligue 1", "uefa", "copa ",
        "exact score:", "correct score:", "round of 16", "knockout stage",
        "leading at halftime", "lead at halftime", "draw at halftime",
        "end in a draw", "score first", "both teams to score", "total corners",
    )):
        return "football"
    return "other"


def _is_clear_sports_result(text: str) -> bool:
    if text.lstrip().startswith(("exact score:", "correct score:")):
        return True
    if any(phrase in text for phrase in ("leading at halftime", "lead at halftime")):
        return True
    if not re.search(r"\bwill\s+.+?\s+win\s+on\s+\d{4}-\d{2}-\d{2}\b", text):
        return False
    return not any(
        word in text
        for word in ("election", "primary", "presidential", "governor", "senate")
    )


def _looks_like_matchup(text: str) -> bool:
    if re.search(r"\bvs\.?\b", text):
        return True
    return _is_clear_sports_result(text)
