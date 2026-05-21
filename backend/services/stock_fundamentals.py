"""
Stock Fundamentals Service
===========================
Fetches comprehensive fundamental data, financials, analyst data,
institutional holders, insider transactions, and options data
from Yahoo Finance via yfinance.

Redis caching strategy:
- Company info / fundamentals: 6h (changes rarely)
- Financials (income, balance, cashflow): 24h (quarterly updates)
- Analyst recommendations: 1h (can change intraday)
- Insider transactions: 6h
- Options chain: 15min (IV changes frequently)
- Earnings dates: 6h
"""
import logging
import time
from typing import Optional, Any
import numpy as np
import pandas as pd

from services.redis_cache import cache_get, cache_set, get_redis

logger = logging.getLogger(__name__)

# ── TTL constants ─────────────────────────────────────────────────────────────
TTL_INFO        = 21600   # 6h  — company info, valuation
TTL_FINANCIALS  = 86400   # 24h — income/balance/cashflow
TTL_ANALYST     = 3600    # 1h  — recommendations, price targets
TTL_INSIDER     = 21600   # 6h  — insider transactions
TTL_OPTIONS     = 900     # 15m — options chain (IV sensitive)
TTL_EARNINGS    = 21600   # 6h  — earnings dates/history
TTL_HOLDERS     = 21600   # 6h  — institutional holders
TTL_NEWS        = 900     # 15m — news (fresh content)
TTL_ESG         = 43200   # 12h — ESG scores (rarely updated)


def _safe(val):
    """Convert numpy/pandas scalars to Python native types, None for NaN/inf."""
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    return val


def _df_to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts."""
    if df is None or df.empty:
        return []
    df = df.copy()
    # Convert index to column if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime('%Y-%m-%d')
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return df.reset_index().to_dict(orient='records')


def _cached(key: str, ttl: int, fetch_fn):
    """Generic cache-aside helper."""
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        result = fetch_fn()
        if result is not None:
            cache_set(key, result, ttl)
        return result
    except Exception as e:
        logger.warning(f"Fundamentals fetch error [{key}]: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_company_overview(symbol: str) -> Optional[dict]:
    """
    Core company info + valuation metrics.
    Returns: name, sector, industry, CEO, employees, market cap,
             beta, PE, forward PE, EPS, book value, dividend yield,
             debt/equity, ROE, ROA, current ratio, quick ratio, FCF,
             revenue growth, earnings growth.
    """
    key = f"fundamentals:overview:{symbol.upper()}"
    return _cached(key, TTL_INFO, lambda: _fetch_overview(symbol))


def get_financials(symbol: str) -> Optional[dict]:
    """
    Income statement, balance sheet, cash flow — annual + quarterly.
    """
    key = f"fundamentals:financials:{symbol.upper()}"
    return _cached(key, TTL_FINANCIALS, lambda: _fetch_financials(symbol))


def get_analyst_data(symbol: str) -> Optional[dict]:
    """
    Analyst recommendations history + price targets.
    """
    key = f"fundamentals:analyst:{symbol.upper()}"
    return _cached(key, TTL_ANALYST, lambda: _fetch_analyst(symbol))


def get_holders(symbol: str) -> Optional[dict]:
    """
    Institutional holders + major holders.
    """
    key = f"fundamentals:holders:{symbol.upper()}"
    return _cached(key, TTL_HOLDERS, lambda: _fetch_holders(symbol))


def get_insider_transactions(symbol: str) -> Optional[list]:
    """
    Recent insider buy/sell transactions.
    """
    key = f"fundamentals:insider:{symbol.upper()}"
    return _cached(key, TTL_INSIDER, lambda: _fetch_insider(symbol))


def get_earnings(symbol: str) -> Optional[dict]:
    """
    Historical EPS + upcoming earnings date.
    """
    key = f"fundamentals:earnings:{symbol.upper()}"
    return _cached(key, TTL_EARNINGS, lambda: _fetch_earnings(symbol))


def get_options_summary(symbol: str) -> Optional[dict]:
    """
    Options chain summary: put/call ratio, max pain, IV, open interest.
    Uses nearest expiration with sufficient liquidity.
    """
    key = f"fundamentals:options:{symbol.upper()}"
    return _cached(key, TTL_OPTIONS, lambda: _fetch_options(symbol))


def get_news(symbol: str) -> Optional[list]:
    """
    Recent news articles with rule-based sentiment scoring.
    TTL: 15min — news is time-sensitive.
    """
    key = f"fundamentals:news:{symbol.upper()}"
    return _cached(key, TTL_NEWS, lambda: _fetch_news(symbol))


def get_esg(symbol: str) -> Optional[dict]:
    """
    ESG sustainability scores (environmental, social, governance).
    Not available for all symbols — returns None gracefully.
    TTL: 12h — ESG scores update infrequently.
    """
    key = f"fundamentals:esg:{symbol.upper()}"
    return _cached(key, TTL_ESG, lambda: _fetch_esg(symbol))


def get_full_fundamentals(symbol: str) -> dict:
    """
    Aggregate all fundamental data in one call.
    Each section is fetched independently so partial failures don't break the whole response.
    """
    sym = symbol.upper()
    t0 = time.time()

    overview     = get_company_overview(sym)
    financials   = get_financials(sym)
    analyst      = get_analyst_data(sym)
    holders      = get_holders(sym)
    insider      = get_insider_transactions(sym)
    earnings     = get_earnings(sym)
    options      = get_options_summary(sym)
    news         = get_news(sym)
    esg          = get_esg(sym)

    return {
        "symbol": sym,
        "overview": overview,
        "financials": financials,
        "analyst": analyst,
        "holders": holders,
        "insider": insider,
        "earnings": earnings,
        "options": options,
        "news": news,
        "esg": esg,
        "_response_time_ms": round((time.time() - t0) * 1000, 1),
    }


# ── Private fetch functions ───────────────────────────────────────────────────

def _fetch_overview(symbol: str) -> dict:
    import yfinance as yf
    stock = yf.Ticker(symbol)
    info = stock.info or {}

    def g(k):
        return _safe(info.get(k))

    return {
        # Identity
        "name":             g("longName") or g("shortName"),
        "sector":           g("sector"),
        "industry":         g("industry"),
        "country":          g("country"),
        "website":          g("website"),
        "description":      (info.get("longBusinessSummary") or "")[:500],
        "ceo":              _extract_ceo(info),
        "employees":        g("fullTimeEmployees"),
        "exchange":         g("exchange"),
        "currency":         g("currency"),
        # Valuation
        "market_cap":       g("marketCap"),
        "enterprise_value": g("enterpriseValue"),
        "beta":             g("beta"),
        "pe_ratio":         g("trailingPE"),
        "forward_pe":       g("forwardPE"),
        "peg_ratio":        g("pegRatio"),
        "price_to_book":    g("priceToBook"),
        "price_to_sales":   g("priceToSalesTrailing12Months"),
        "ev_to_ebitda":     g("enterpriseToEbitda"),
        "ev_to_revenue":    g("enterpriseToRevenue"),
        # Per-share
        "eps":              g("trailingEps"),
        "forward_eps":      g("forwardEps"),
        "book_value":       g("bookValue"),
        # Dividends
        "dividend_yield":   g("dividendYield"),
        "dividend_rate":    g("dividendRate"),
        "payout_ratio":     g("payoutRatio"),
        "ex_dividend_date": g("exDividendDate"),
        # Profitability
        "profit_margin":    g("profitMargins"),
        "operating_margin": g("operatingMargins"),
        "gross_margin":     g("grossMargins"),
        "roe":              g("returnOnEquity"),
        "roa":              g("returnOnAssets"),
        "roic":             g("returnOnCapital"),
        # Leverage
        "debt_to_equity":   g("debtToEquity"),
        "current_ratio":    g("currentRatio"),
        "quick_ratio":      g("quickRatio"),
        # Growth
        "revenue_growth":   g("revenueGrowth"),
        "earnings_growth":  g("earningsGrowth"),
        "earnings_quarterly_growth": g("earningsQuarterlyGrowth"),
        # Cash flow
        "free_cashflow":    g("freeCashflow"),
        "operating_cashflow": g("operatingCashflow"),
        # Revenue / earnings
        "total_revenue":    g("totalRevenue"),
        "revenue_per_share": g("revenuePerShare"),
        "ebitda":           g("ebitda"),
        # Float / shares
        "shares_outstanding": g("sharesOutstanding"),
        "float_shares":     g("floatShares"),
        "shares_short":     g("sharesShort"),
        "short_ratio":      g("shortRatio"),
        "short_percent_float": g("shortPercentOfFloat"),
        # 52-week
        "52w_high":         g("fiftyTwoWeekHigh"),
        "52w_low":          g("fiftyTwoWeekLow"),
        "50d_avg":          g("fiftyDayAverage"),
        "200d_avg":         g("twoHundredDayAverage"),
        # Analyst consensus
        "analyst_rating":   g("recommendationKey"),
        "target_mean":      g("targetMeanPrice"),
        "target_high":      g("targetHighPrice"),
        "target_low":       g("targetLowPrice"),
        "num_analysts":     g("numberOfAnalystOpinions"),
    }


def _extract_ceo(info: dict) -> Optional[str]:
    """Extract CEO name from companyOfficers list."""
    officers = info.get("companyOfficers") or []
    for o in officers:
        title = (o.get("title") or "").upper()
        if "CEO" in title or "CHIEF EXECUTIVE" in title:
            return o.get("name")
    return None


def _fetch_financials(symbol: str) -> dict:
    import yfinance as yf
    stock = yf.Ticker(symbol)

    def safe_df(df):
        if df is None or df.empty:
            return []
        df = df.copy()
        df.columns = [str(c)[:10] for c in df.columns]  # shorten date strings
        df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        return df.reset_index().rename(columns={"index": "metric"}).to_dict(orient="records")

    # Annual
    income_annual  = safe_df(stock.financials)
    balance_annual = safe_df(stock.balance_sheet)
    cashflow_annual = safe_df(stock.cashflow)

    # Quarterly
    income_q  = safe_df(stock.quarterly_financials)
    balance_q = safe_df(stock.quarterly_balance_sheet)
    cashflow_q = safe_df(stock.quarterly_cashflow)

    return {
        "income_annual":   income_annual,
        "balance_annual":  balance_annual,
        "cashflow_annual": cashflow_annual,
        "income_quarterly":   income_q,
        "balance_quarterly":  balance_q,
        "cashflow_quarterly": cashflow_q,
    }


def _fetch_analyst(symbol: str) -> dict:
    import yfinance as yf
    stock = yf.Ticker(symbol)

    # Recommendations history
    recs = []
    try:
        df = stock.recommendations
        if df is not None and not df.empty:
            df = df.tail(50).copy()
            df = df.replace({np.nan: None})
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.strftime('%Y-%m-%d')
            recs = df.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")
    except Exception:
        pass

    # Recommendations summary (buy/hold/sell counts)
    rec_summary = {}
    try:
        df = stock.recommendations_summary
        if df is not None and not df.empty:
            df = df.replace({np.nan: None})
            rec_summary = df.to_dict(orient="records")
    except Exception:
        pass

    # Price targets
    targets = {}
    try:
        pt = stock.analyst_price_targets
        if pt is not None:
            targets = {k: _safe(v) for k, v in pt.items()}
    except Exception:
        pass

    # Upgrades/downgrades
    upgrades = []
    try:
        df = stock.upgrades_downgrades
        if df is not None and not df.empty:
            df = df.tail(20).copy()
            df = df.replace({np.nan: None})
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.strftime('%Y-%m-%d')
            upgrades = df.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")
    except Exception:
        pass

    return {
        "recommendations": recs,
        "recommendations_summary": rec_summary,
        "price_targets": targets,
        "upgrades_downgrades": upgrades,
    }


def _fetch_holders(symbol: str) -> dict:
    import yfinance as yf
    stock = yf.Ticker(symbol)

    institutional = []
    try:
        df = stock.institutional_holders
        if df is not None and not df.empty:
            df = df.replace({np.nan: None})
            institutional = df.to_dict(orient="records")
    except Exception:
        pass

    major = []
    try:
        df = stock.major_holders
        if df is not None and not df.empty:
            df = df.replace({np.nan: None})
            major = df.to_dict(orient="records")
    except Exception:
        pass

    mutual_funds = []
    try:
        df = stock.mutualfund_holders
        if df is not None and not df.empty:
            df = df.replace({np.nan: None})
            mutual_funds = df.to_dict(orient="records")
    except Exception:
        pass

    return {
        "institutional": institutional[:15],  # top 15
        "major": major,
        "mutual_funds": mutual_funds[:10],
    }


def _fetch_insider(symbol: str) -> list:
    import yfinance as yf
    stock = yf.Ticker(symbol)
    try:
        df = stock.insider_transactions
        if df is None or df.empty:
            return []
        df = df.head(30).copy()
        df = df.replace({np.nan: None})
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.strftime('%Y-%m-%d')
        return df.reset_index().rename(columns={"index": "date"}).to_dict(orient="records")
    except Exception as e:
        logger.warning(f"Insider fetch failed for {symbol}: {e}")
        return []


def _fetch_earnings(symbol: str) -> dict:
    import yfinance as yf
    stock = yf.Ticker(symbol)

    # Historical EPS (annual)
    eps_annual = []
    try:
        df = stock.earnings
        if df is not None and not df.empty:
            df = df.replace({np.nan: None})
            eps_annual = df.reset_index().to_dict(orient="records")
    except Exception:
        pass

    # Quarterly EPS
    eps_quarterly = []
    try:
        df = stock.quarterly_earnings
        if df is not None and not df.empty:
            df = df.replace({np.nan: None})
            eps_quarterly = df.reset_index().to_dict(orient="records")
    except Exception:
        pass

    # Upcoming earnings date
    next_earnings = None
    try:
        cal = stock.calendar
        if cal is not None:
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    next_earnings = str(ed[0]) if isinstance(ed, list) else str(ed)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                ed = cal.get("Earnings Date")
                if ed is not None:
                    next_earnings = str(ed.iloc[0]) if hasattr(ed, 'iloc') else str(ed)
    except Exception:
        pass

    return {
        "eps_annual": eps_annual,
        "eps_quarterly": eps_quarterly,
        "next_earnings_date": next_earnings,
    }


def _fetch_options(symbol: str) -> dict:
    import yfinance as yf
    stock = yf.Ticker(symbol)

    try:
        expirations = stock.options
        if not expirations:
            return {"available": False}

        # Pick nearest expiration with at least 7 days out
        from datetime import datetime, timedelta
        today = datetime.today()
        target_exp = None
        for exp in expirations:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                if (exp_dt - today).days >= 7:
                    target_exp = exp
                    break
            except Exception:
                continue

        if not target_exp:
            target_exp = expirations[0]

        chain = stock.option_chain(target_exp)
        calls = chain.calls
        puts  = chain.puts

        # Put/Call ratio (by open interest)
        total_call_oi = float(calls["openInterest"].sum()) if "openInterest" in calls else 0
        total_put_oi  = float(puts["openInterest"].sum())  if "openInterest" in puts  else 0
        pc_ratio = round(total_put_oi / max(total_call_oi, 1), 3)

        # Max pain: strike where total option value is minimized
        max_pain_strike = _calc_max_pain(calls, puts)

        # IV summary
        call_iv_mean = _safe(calls["impliedVolatility"].mean()) if "impliedVolatility" in calls else None
        put_iv_mean  = _safe(puts["impliedVolatility"].mean())  if "impliedVolatility" in puts  else None

        # Top strikes by open interest
        top_calls = []
        top_puts  = []
        if not calls.empty and "openInterest" in calls:
            top_calls = calls.nlargest(5, "openInterest")[
                ["strike", "lastPrice", "impliedVolatility", "openInterest", "volume"]
            ].replace({np.nan: None}).to_dict(orient="records")
        if not puts.empty and "openInterest" in puts:
            top_puts = puts.nlargest(5, "openInterest")[
                ["strike", "lastPrice", "impliedVolatility", "openInterest", "volume"]
            ].replace({np.nan: None}).to_dict(orient="records")

        return {
            "available": True,
            "expiration": target_exp,
            "all_expirations": list(expirations[:8]),
            "put_call_ratio": pc_ratio,
            "max_pain": max_pain_strike,
            "call_iv_mean": call_iv_mean,
            "put_iv_mean": put_iv_mean,
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "top_calls": top_calls,
            "top_puts": top_puts,
        }

    except Exception as e:
        logger.warning(f"Options fetch failed for {symbol}: {e}")
        return {"available": False, "error": str(e)}


def _calc_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
    """Calculate max pain strike price."""
    try:
        all_strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
        if not all_strikes:
            return None

        min_pain = float("inf")
        max_pain_strike = all_strikes[0]

        for strike in all_strikes:
            call_pain = float(((calls["strike"] - strike).clip(lower=0) * calls["openInterest"]).sum())
            put_pain  = float(((strike - puts["strike"]).clip(lower=0) * puts["openInterest"]).sum())
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                max_pain_strike = strike

        return float(max_pain_strike)
    except Exception:
        return None


# ── News & Sentiment ──────────────────────────────────────────────────────────

# Positive / negative keyword sets for rule-based sentiment
_POS_WORDS = frozenset([
    "beat", "beats", "record", "growth", "profit", "surge", "rally", "upgrade",
    "strong", "positive", "gain", "rise", "rises", "raised", "raises", "buy",
    "outperform", "exceed", "exceeds", "bullish", "boost", "boosts", "win",
    "wins", "award", "partnership", "deal", "acquisition", "dividend", "buyback",
])
_NEG_WORDS = frozenset([
    "miss", "misses", "loss", "losses", "decline", "falls", "fell", "cut",
    "cuts", "downgrade", "weak", "negative", "drop", "drops", "sell", "lawsuit",
    "investigation", "fraud", "recall", "layoff", "layoffs", "bankruptcy",
    "default", "warning", "concern", "risk", "bearish", "disappoints",
])

# Category keywords for event tagging
_CATEGORY_MAP = {
    "earnings":     ["earnings", "eps", "revenue", "quarterly", "annual results", "profit"],
    "analyst":      ["upgrade", "downgrade", "price target", "rating", "analyst", "outperform"],
    "m&a":          ["acquisition", "merger", "takeover", "deal", "buys", "acquires"],
    "dividend":     ["dividend", "buyback", "repurchase", "yield"],
    "legal":        ["lawsuit", "investigation", "sec", "fraud", "settlement", "fine"],
    "product":      ["launch", "product", "release", "partnership", "contract", "award"],
    "macro":        ["fed", "interest rate", "inflation", "gdp", "economy", "recession"],
}


def _score_sentiment(title: str, summary: str = "") -> dict:
    """
    Rule-based sentiment scoring on title + summary text.
    Returns: score (-1.0 to 1.0), label, confidence.
    """
    text = (title + " " + summary).lower()
    words = set(text.split())

    pos_hits = len(words & _POS_WORDS)
    neg_hits = len(words & _NEG_WORDS)
    total = pos_hits + neg_hits

    if total == 0:
        return {"score": 0.0, "label": "neutral", "confidence": "low"}

    score = (pos_hits - neg_hits) / total
    confidence = "high" if total >= 3 else "medium" if total >= 2 else "low"

    if score > 0.3:
        label = "positive"
    elif score < -0.3:
        label = "negative"
    else:
        label = "neutral"

    return {"score": round(score, 3), "label": label, "confidence": confidence}


def _tag_category(title: str) -> str:
    """Tag news article with a category based on title keywords."""
    t = title.lower()
    for cat, keywords in _CATEGORY_MAP.items():
        if any(kw in t for kw in keywords):
            return cat
    return "general"


def _fetch_news(symbol: str) -> list:
    """
    Fetch recent news from yfinance and enrich with:
    - Rule-based sentiment (positive/neutral/negative)
    - Category tag (earnings, analyst, m&a, dividend, legal, product, macro)
    - Relative time label

    Handles both old (flat) and new (nested content) yfinance news formats.
    """
    import yfinance as yf
    from datetime import datetime, timezone

    stock = yf.Ticker(symbol)
    try:
        raw_news = stock.news or []
    except Exception as e:
        logger.warning(f"News fetch failed for {symbol}: {e}")
        return []

    now_ts = int(datetime.now(timezone.utc).timestamp())
    articles = []

    for item in raw_news[:30]:
        try:
            # ── New format: item has a 'content' sub-dict ──
            content = item.get("content") or {}
            if content:
                title     = content.get("title", "")
                summary   = content.get("summary", "") or content.get("description", "")
                publisher = (content.get("provider") or {}).get("displayName", "")
                link      = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url", "")
                pub_str   = content.get("pubDate", "")
                # Parse ISO date string → unix timestamp
                pub_ts = 0
                if pub_str:
                    try:
                        from datetime import timezone as tz
                        dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        pub_ts = int(dt.timestamp())
                    except Exception:
                        pass
                # Thumbnail
                thumbnail = None
                thumb = content.get("thumbnail") or {}
                resolutions = thumb.get("resolutions", [])
                # Pick smallest thumbnail for performance
                for res in sorted(resolutions, key=lambda r: r.get("width", 9999)):
                    if res.get("url"):
                        thumbnail = res["url"]
                        break
            else:
                # ── Old flat format ──
                title     = item.get("title", "")
                summary   = ""
                publisher = item.get("publisher", "")
                link      = item.get("link", "")
                pub_ts    = item.get("providerPublishTime", 0)
                thumbnail = None
                thumb_data = item.get("thumbnail")
                if thumb_data and isinstance(thumb_data, dict):
                    resolutions = thumb_data.get("resolutions", [])
                    if resolutions:
                        thumbnail = resolutions[0].get("url")

            if not title:
                continue

            # Relative time
            age_sec = now_ts - pub_ts if pub_ts else None
            if age_sec is not None:
                if age_sec < 3600:
                    rel_time = f"{max(1, age_sec // 60)}m ago"
                elif age_sec < 86400:
                    rel_time = f"{age_sec // 3600}h ago"
                else:
                    rel_time = f"{age_sec // 86400}d ago"
            else:
                rel_time = None

            sentiment = _score_sentiment(title, summary)
            category  = _tag_category(title + " " + summary)

            articles.append({
                "title":     title,
                "summary":   (summary or "")[:200],
                "publisher": publisher,
                "link":      link,
                "published": pub_ts,
                "rel_time":  rel_time,
                "thumbnail": thumbnail,
                "sentiment": sentiment,
                "category":  category,
            })
        except Exception:
            continue

    articles.sort(key=lambda x: x.get("published", 0), reverse=True)

    labels = [a["sentiment"]["label"] for a in articles]
    pos = labels.count("positive")
    neg = labels.count("negative")
    neu = labels.count("neutral")
    total = len(labels)
    overall = "positive" if pos > neg and pos > neu else "negative" if neg > pos and neg > neu else "neutral"
    overall_score = round((pos - neg) / max(total, 1), 3)

    return {
        "articles": articles,
        "summary": {
            "total": total,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "overall_sentiment": overall,
            "overall_score": overall_score,
        },
    }


# ── ESG / Sustainability ──────────────────────────────────────────────────────

def _fetch_esg(symbol: str) -> Optional[dict]:
    """
    Fetch ESG sustainability scores.
    Yahoo Finance's dedicated sustainability endpoint is deprecated (returns 404).
    We extract available ESG fields from the main info dict instead.
    Returns None if no ESG data is available.
    """
    import yfinance as yf

    stock = yf.Ticker(symbol)
    try:
        info = stock.info or {}

        # ESG fields that may be present in info
        total_score = _safe(info.get("esgScores") or info.get("totalEsg"))
        env_score   = _safe(info.get("environmentScore"))
        soc_score   = _safe(info.get("socialScore"))
        gov_score   = _safe(info.get("governanceScore"))

        # If none of the core scores are available, return None
        if all(v is None for v in [total_score, env_score, soc_score, gov_score]):
            # Try the sustainability DataFrame as a last resort
            try:
                sus = stock.sustainability
                if sus is not None and not sus.empty:
                    col = sus.columns[0] if len(sus.columns) > 0 else None
                    if col:
                        d = sus[col].to_dict()
                        total_score = _safe(d.get("totalEsg") or d.get("esgScores"))
                        env_score   = _safe(d.get("environmentScore"))
                        soc_score   = _safe(d.get("socialScore"))
                        gov_score   = _safe(d.get("governanceScore"))
            except Exception:
                pass

        if all(v is None for v in [total_score, env_score, soc_score, gov_score]):
            return None

        def g(k):
            v = info.get(k)
            return _safe(v)

        def risk_level(score):
            if score is None:
                return None
            if score < 10:   return "negligible"
            if score < 20:   return "low"
            if score < 30:   return "medium"
            if score < 40:   return "high"
            return "severe"

        return {
            "available": True,
            "total_score":       total_score,
            "total_risk_level":  risk_level(total_score),
            "percentile":        g("percentile"),
            "peer_group":        g("peerGroup"),
            "peer_count":        g("peerCount"),
            "environment_score": env_score,
            "social_score":      soc_score,
            "governance_score":  gov_score,
            "environment_risk":  risk_level(env_score),
            "social_risk":       risk_level(soc_score),
            "governance_risk":   risk_level(gov_score),
            "controversy_level": g("highestControversy"),
            "controversy_score": g("controversyScore"),
            "adult":             g("adult"),
            "alcoholic":         g("alcoholic"),
            "animal_testing":    g("animalTesting"),
            "catholic":          g("catholic"),
            "controversial_weapons": g("controversialWeapons"),
            "small_arms":        g("smallArms"),
            "fur_leather":       g("furLeather"),
            "gambling":          g("gambling"),
            "gmo":               g("gmo"),
            "military_contract": g("militaryContract"),
            "nuclear":           g("nuclear"),
            "pesticides":        g("pesticides"),
            "palm_oil":          g("palmOil"),
            "coal":              g("coal"),
            "tobacco":           g("tobacco"),
            "rating_year":       g("ratingYear"),
            "rating_month":      g("ratingMonth"),
        }

    except Exception as e:
        logger.warning(f"ESG fetch failed for {symbol}: {e}")
        return None
