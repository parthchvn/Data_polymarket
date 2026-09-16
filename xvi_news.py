#!/usr/bin/env python3
"""
xvi_news.py - time-indexed news for the selected markets, joined onto decisions without leakage.

Sources
  gdelt   GDELT DOC 2.0 API: article list per query and time window, with `seendate` = when GDELT first
          crawled the article (15-minute resolution). This is an availability time.
  rss     Google News RSS search with `after:`/`before:` day operators. Returns at most 100 items per
          request, and for windowed searches the pubDate is usually a day-level placeholder
          (07:00:00 GMT), so it is a publication DATE, not a time.
  wayback (optional, per article) Internet Archive CDX first-capture time: an upper bound on availability.

Timestamps kept per article: published_at (publisher/aggregator claim), first_seen_at (crawl/capture),
and availability_at = first_seen_at if known, else published_at rounded to the END of its UTC day
(conservative: a day-dated article is only assumed public once the day is over). The as-of join uses
availability_at strictly less than the decision timestamp.

Commands
  python xvi_news.py collect  [--markets ...] [--sources rss,gdelt] [--window-days 1] [--out data/news]
  python xvi_news.py dedupe   [--out data/news]
  python xvi_news.py link     [--out data/news]              # relevance score per (market, article)
  python xvi_news.py asof     [--out data/news] [--scored data/scored] [--min-relevance 0.15]
  python xvi_news.py coverage [--out data/news]
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse as up
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = "xvi-news/0.1 (research; contact via repo)"
DEFAULT_QUERIES = {  # market_id -> query phrases; anything not listed falls back to the market question
    "507300": ["Inter Milan Champions League", "Inter PSG Champions League final"],
    "2446852": ["Ronaldo World Cup", "Cristiano Ronaldo cry World Cup"],
    "542539": ["Fed September meeting interest rates", "FOMC September decision"],
    "570360": ["Fed December meeting rate cut", "FOMC December decision"],
}
STOP = set("will the a an of in on at by to for and or is are be after before than more less bps meeting rates rate does do".split())
DAY_PLACEHOLDER_HOURS = {7, 8}  # Google's windowed RSS reports 07:00:00 GMT (08:00 in DST) when it has only a date


def utc_now() -> str: return dt.datetime.now(dt.timezone.utc).isoformat()
def epoch(d: dt.datetime) -> int: return int(d.timestamp())


def session() -> requests.Session:
    s = requests.Session(); s.headers["User-Agent"] = UA; return s


def queries_for(market: dict, cfg: dict) -> list[str]:
    if market["id"] in cfg: return cfg[market["id"]]
    words = [w for w in re.findall(r"[A-Za-z0-9']+", market["question"]) if w.lower() not in STOP]
    return [" ".join(words[:6])]


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def rss_window(s: requests.Session, query: str, day0: dt.date, day1: dt.date) -> list[dict]:
    q = f"{query} after:{day0:%Y-%m-%d} before:{day1:%Y-%m-%d}"
    u = "https://news.google.com/rss/search?" + up.urlencode({"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    for attempt in range(4):
        r = s.get(u, timeout=60)
        if r.status_code == 200: break
        time.sleep(3 * (attempt + 1))
    else:
        return [{"_error": f"rss HTTP {r.status_code}"}]
    out = []
    for it in ET.fromstring(r.content).findall(".//item"):
        pub = it.findtext("pubDate"); ts = None; res = "unknown"
        if pub:
            d = email.utils.parsedate_to_datetime(pub).astimezone(dt.timezone.utc)
            ts = epoch(d); res = "day" if (d.hour in DAY_PLACEHOLDER_HOURS and d.minute == 0 and d.second == 0) else "second"
        src = it.find("source")
        out.append({"source": "rss", "query": query, "title": (it.findtext("title") or "").strip(), "url": it.findtext("link"),
                    "publisher": src.text if src is not None else None, "published_at": ts, "published_resolution": res,
                    "first_seen_at": None, "raw_pubdate": pub})
    return out


def gdelt_window(s: requests.Session, query: str, t0: dt.datetime, t1: dt.datetime) -> list[dict]:
    q = " ".join(f'"{p}"' if " " in p else p for p in [query]) + " sourcelang:english"
    u = "https://api.gdeltproject.org/api/v2/doc/doc?" + up.urlencode({"query": q, "mode": "artlist", "format": "json", "maxrecords": 250, "sort": "DateAsc",
                                                                        "startdatetime": t0.strftime("%Y%m%d%H%M%S"), "enddatetime": t1.strftime("%Y%m%d%H%M%S")})
    for attempt in range(3):
        try:
            r = s.get(u, timeout=90)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                out = []
                for a in r.json().get("articles", []):
                    seen = epoch(dt.datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc))
                    out.append({"source": "gdelt", "query": query, "title": a.get("title", "").strip(), "url": a.get("url"), "publisher": a.get("domain"),
                                "published_at": None, "published_resolution": "unknown", "first_seen_at": seen, "language": a.get("language"), "country": a.get("sourcecountry")})
                return out
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return [{"_error": f"gdelt unavailable ({r.status_code if 'r' in dir() else 'no response'})"}]


def cmd_collect(args):
    sel, out = Path(args.selected), Path(args.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    markets = json.loads((sel / "markets.json").read_text())
    if args.markets: markets = [m for m in markets if m["id"] in args.markets]
    cfg_path = out / "queries.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else DEFAULT_QUERIES
    cfg_path.write_text(json.dumps(cfg, indent=2))
    settle_dir = Path(args.settlement)
    s = session(); sources = args.sources.split(",")
    log = {"started_at": utc_now(), "sources": sources, "window_days": args.window_days, "markets": {}}
    for m in markets:
        mid = m["id"]
        st = settle_dir / f"{m['condition_id']}.json"
        t_end = json.loads(st.read_text()).get("settlement_timestamp") if st.exists() else None
        trades = pd.read_parquet(sel / "trades.parquet", columns=["market_id", "timestamp"])
        tr = trades[trades.market_id.eq(mid)]
        t0 = dt.datetime.fromtimestamp(int(tr.timestamp.min()), dt.timezone.utc) - dt.timedelta(days=args.lead_days)
        t1 = dt.datetime.fromtimestamp(int(t_end or tr.timestamp.max()), dt.timezone.utc) + dt.timedelta(days=1)
        qs = queries_for(m, cfg); rows = []; saturated = 0; reqs = 0; errors = 0
        fh = open(out / "raw" / f"{mid}.jsonl", "a")
        d = t0.date()
        while d <= t1.date():
            d2 = min(d + dt.timedelta(days=args.window_days), t1.date() + dt.timedelta(days=1))
            for q in qs:
                for src in sources:
                    if src == "rss": batch = rss_window(s, q, d, d2)
                    elif src == "gdelt":
                        batch = gdelt_window(s, q, dt.datetime.combine(d, dt.time(), dt.timezone.utc), dt.datetime.combine(d2, dt.time(), dt.timezone.utc))
                    else: continue
                    reqs += 1
                    if batch and "_error" in batch[0]: errors += 1; continue
                    if (src == "rss" and len(batch) >= 100) or (src == "gdelt" and len(batch) >= 250): saturated += 1
                    for b in batch:
                        b.update(market_id=mid, window_start=str(d), window_end=str(d2), fetched_at=utc_now(), saturated_window=(len(batch) >= (100 if src == "rss" else 250)))
                        fh.write(json.dumps(b) + "\n"); rows.append(b)
                    time.sleep(args.pause + random.random() * 0.5)
            d = d2
        fh.close()
        log["markets"][mid] = {"question": m["question"], "queries": qs, "from": str(t0.date()), "to": str(t1.date()), "requests": reqs,
                               "saturated_windows": saturated, "errors": errors, "rows": len(rows)}
        print(f"{mid}: {len(rows):,} rows from {reqs} requests, {saturated} saturated windows, {errors} errors  [{t0.date()} .. {t1.date()}]", flush=True)
    (out / "collect_log.json").write_text(json.dumps(log, indent=2))


# --------------------------------------------------------------------------- #
# dedupe
# --------------------------------------------------------------------------- #
def norm_title(t: str) -> str:
    t = re.sub(r"\s+-\s+[^-]+$", "", t or "")            # strip trailing " - Publisher" that Google appends
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def canon_url(u: str | None) -> str | None:
    if not u: return None
    p = up.urlsplit(u); qs = [(k, v) for k, v in up.parse_qsl(p.query) if not k.lower().startswith(("utm_", "fbclid", "gclid"))]
    return up.urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), up.urlencode(qs), ""))


def cmd_dedupe(args):
    out = Path(args.out); rows = []
    for f in sorted((out / "raw").glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df.title.notna() & (df.title != "")]
    df["url_canon"] = df["url"].map(canon_url); df["title_norm"] = df["title"].map(norm_title)
    df["pub_day"] = pd.to_datetime(df["published_at"].fillna(df["first_seen_at"]), unit="s", utc=True).dt.date.astype(str)
    df["article_key"] = np.where(df["source"].eq("gdelt"), df["url_canon"], df["title_norm"] + "|" + df["publisher"].fillna("") + "|" + df["pub_day"])
    df["article_id"] = df["article_key"].map(lambda k: hashlib.sha1(k.encode()).hexdigest()[:16])
    # one article row: earliest availability evidence wins; keep the set of (market, query) hits separately
    df = df.sort_values(["first_seen_at", "published_at"], na_position="last")
    hits = df[["article_id", "market_id", "query", "source", "window_start", "saturated_window"]].drop_duplicates()
    art = df.drop_duplicates("article_id").copy()
    art["availability_at"] = art["first_seen_at"]
    pub = art["published_at"].astype("float")
    day_end = pub - (pub % 86400) + 86399                       # 23:59:59 UTC of the publication date (epoch arithmetic)
    same_second = art["published_resolution"].eq("second")
    art["availability_at"] = art["availability_at"].astype("float")
    art.loc[art["availability_at"].isna() & same_second, "availability_at"] = pub[art["availability_at"].isna() & same_second]
    art.loc[art["availability_at"].isna(), "availability_at"] = day_end[art["availability_at"].isna()]
    art["availability_at"] = art["availability_at"].astype("Int64")
    art["availability_basis"] = np.where(art["first_seen_at"].notna(), "first_seen", np.where(art["published_resolution"].eq("second"), "published_time", "published_day_end"))
    cols = ["article_id", "title", "publisher", "url", "url_canon", "source", "published_at", "published_resolution", "first_seen_at", "availability_at", "availability_basis", "fetched_at"]
    art[cols].sort_values("availability_at").to_parquet(out / "news_articles.parquet", index=False)
    hits.to_parquet(out / "news_hits.parquet", index=False)
    print(f"{len(df):,} raw rows -> {len(art):,} articles ({(art.availability_basis=='first_seen').sum():,} with a crawl/capture time, "
          f"{(art.availability_basis=='published_day_end').sum():,} day-dated only); {len(hits):,} (market, query) hits")


# --------------------------------------------------------------------------- #
# link: relevance of each article to each market it was retrieved for
# --------------------------------------------------------------------------- #
def cmd_link(args):
    out, sel = Path(args.out), Path(args.selected)
    art = pd.read_parquet(out / "news_articles.parquet"); hits = pd.read_parquet(out / "news_hits.parquet")
    markets = {m["id"]: m for m in json.loads((sel / "markets.json").read_text())}
    cfg = json.loads((out / "queries.json").read_text())
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    pairs = hits[["article_id", "market_id"]].drop_duplicates().merge(art[["article_id", "title"]], on="article_id")
    recs = []
    for mid, g in pairs.groupby("market_id"):
        m = markets[mid]; ref = " ".join([m["question"], m.get("event_title") or "", " ".join(queries_for(m, cfg))])
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit(list(g["title"]) + [ref])
        sim = cosine_similarity(vec.transform(g["title"]), vec.transform([ref])).ravel()
        phrases = [p.lower() for q in queries_for(m, cfg) for p in q.split()]
        kw = g["title"].str.lower().map(lambda t: sum(p in t for p in set(phrases)) / max(len(set(phrases)), 1))
        recs.append(pd.DataFrame({"market_id": mid, "article_id": g["article_id"].values, "relevance": sim, "keyword_share": kw.values, "relevance_method": "tfidf_title_vs_question"}))
    link = pd.concat(recs).sort_values(["market_id", "relevance"], ascending=[True, False])
    link.to_parquet(out / "market_news.parquet", index=False)
    print(link.groupby("market_id").relevance.describe()[["count", "mean", "50%", "75%", "max"]].round(3).to_string())


# --------------------------------------------------------------------------- #
# asof: per-decision news features using availability_at < decision timestamp
# --------------------------------------------------------------------------- #
def cmd_asof(args):
    out, scored = Path(args.out), Path(args.scored)
    art = pd.read_parquet(out / "news_articles.parquet"); link = pd.read_parquet(out / "market_news.parquet")
    keep = (link.relevance >= args.min_relevance) | (link.keyword_share >= args.min_keyword_share)
    link = link[keep].merge(art[["article_id", "availability_at", "availability_basis"]], on="article_id")
    d = pd.read_parquet(scored / "market_decisions.parquet", columns=["decision_id", "market_id", "timestamp"])
    feats = []
    for mid, g in d.groupby("market_id"):
        t = np.sort(link.loc[link.market_id.eq(mid), "availability_at"].astype("int64").to_numpy())
        ts = g["timestamp"].to_numpy()
        n_before = np.searchsorted(t, ts, side="left")          # strictly earlier than the decision
        f = pd.DataFrame({"decision_id": g["decision_id"].values, "news_n_total_before": n_before})
        for name, secs in (("1h", 3600), ("6h", 6 * 3600), ("24h", 86400), ("7d", 7 * 86400)):
            f[f"news_n_{name}"] = n_before - np.searchsorted(t, ts - secs, side="left")
        last = np.where(n_before > 0, t[np.maximum(n_before - 1, 0)], np.nan)
        f["hours_since_last_news"] = (ts - last) / 3600.0
        feats.append(f)
    feats = pd.concat(feats)
    feats.to_parquet(out / "decision_news_features.parquet", index=False)
    basis = link.availability_basis.value_counts().to_dict()
    print(f"{len(feats):,} decisions; linked articles (relevance>={args.min_relevance} or keyword_share>={args.min_keyword_share}): {len(link):,} (availability basis {basis})")
    print(feats[["news_n_1h", "news_n_24h", "news_n_7d", "hours_since_last_news"]].describe().round(2).to_string())
    if basis.get("published_day_end", 0) > basis.get("first_seen", 0):
        print("NOTE: most articles carry only a publication date, so availability is set to the end of that day; "
              "the 1h/6h windows are not meaningful until first-seen times (GDELT or Wayback) are attached.")


def cmd_coverage(args):
    out = Path(args.out)
    art = pd.read_parquet(out / "news_articles.parquet"); hits = pd.read_parquet(out / "news_hits.parquet")
    x = hits.merge(art[["article_id", "availability_at"]], on="article_id")
    x["day"] = pd.to_datetime(x.availability_at, unit="s", utc=True).dt.date
    per_day = x.groupby(["market_id", "day"]).article_id.nunique()
    print(per_day.groupby("market_id").describe()[["count", "mean", "50%", "max"]].rename(columns={"count": "days_with_news"}).round(1).to_string())
    print("\nsaturated (market, query, window) cells, where the 100-item RSS cap was hit and articles were lost:")
    print(hits[hits.saturated_window].groupby(["market_id", "query"]).window_start.nunique().to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("collect"); a.add_argument("--selected", default="data/selected"); a.add_argument("--settlement", default="data/settlement"); a.add_argument("--out", default="data/news")
    a.add_argument("--markets", nargs="*"); a.add_argument("--sources", default="rss,gdelt"); a.add_argument("--window-days", type=int, default=1); a.add_argument("--lead-days", type=int, default=3)
    a.add_argument("--pause", type=float, default=1.0); a.set_defaults(fn=cmd_collect)
    a = sp.add_parser("dedupe"); a.add_argument("--out", default="data/news"); a.set_defaults(fn=cmd_dedupe)
    a = sp.add_parser("link"); a.add_argument("--out", default="data/news"); a.add_argument("--selected", default="data/selected"); a.set_defaults(fn=cmd_link)
    a = sp.add_parser("asof"); a.add_argument("--out", default="data/news"); a.add_argument("--scored", default="data/scored"); a.add_argument("--min-relevance", type=float, default=0.15); a.add_argument("--min-keyword-share", type=float, default=0.5); a.set_defaults(fn=cmd_asof)
    a = sp.add_parser("coverage"); a.add_argument("--out", default="data/news"); a.set_defaults(fn=cmd_coverage)
    args = p.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
