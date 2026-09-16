#!/usr/bin/env python3
"""
xvi_extract.py - pull trades for selected Polymarket market IDs out of
SII-WANGZJ/Polymarket_data without downloading the archive.

Why this works
--------------
trades.parquet is a single Parquet file (37.5 GB, 1.03e9 rows, 1027 row
groups at revision 6d3c336c) whose row groups are sorted by timestamp. Parquet
stores each column of each row group as its own contiguous byte range, and the
Hugging Face CDN honours HTTP Range requests. So:

  index    read only the `market_id` column chunk of every row group
           (about 30 MB in total) and record which row groups contain which
           market, and how many rows. Done once per dataset revision.
  extract  fetch only the row groups that contain the requested markets,
           keep only the matching rows, verify row counts against the index,
           write data/selected/{trades.parquet,markets.csv,markets.json,
           extraction_manifest.json}.

Cost of an extract scales with the number of row groups a market's trades
touch (its active lifetime, roughly 35-45 MB per row group), not with the
archive size.

Commands
--------
  python xvi_extract.py index   --out data/index
  python xvi_extract.py extract --markets 507300 2446852 --index data/index --out data/selected
  python xvi_extract.py extract --markets 507300 2446852 123456 \
        --index data/index --out data/selected --existing /path/to/news_attr/data/selected
  python xvi_extract.py verify  --dir /path/to/news_attr/data/selected
  python xvi_extract.py lookup  --markets 507300 --index data/index

Set HF_TOKEN in the environment for higher rate limits (optional).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

DATASET = "SII-WANGZJ/Polymarket_data"
# Revision the existing news_attr extract was built from; also the HF head on 2026-09-15.
DEFAULT_REVISION = "6d3c336c39cf1a2dfe53d702ad2c110ab5bdbfde"
TRADES_FILE = "trades.parquet"
MARKETS_FILE = "markets.parquet"
MARKET_COLS = [
    "id", "question", "slug", "condition_id", "token1", "token2", "answer1", "answer2",
    "closed", "active", "archived", "outcome_prices", "volume", "event_id", "event_slug",
    "event_title", "created_at", "end_date", "updated_at", "neg_risk",
]
SORT_KEYS = [("timestamp", "ascending"), ("block_number", "ascending"), ("log_index", "ascending")]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Remote file access: one HTTP Range request per read, no full download.
# --------------------------------------------------------------------------- #
class HFSource:
    """Resolves the CDN URL for a file at a pinned revision and reports its size and LFS sha256."""

    def __init__(self, revision: str, filename: str, session: requests.Session | None = None):
        self.revision = revision
        self.filename = filename
        self.resolve_url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{filename}"
        self.session = session or make_session()
        self.size, self.lfs_sha256 = self._stat()

    def _stat(self):
        api = f"https://huggingface.co/api/datasets/{DATASET}/tree/{self.revision}"
        r = self.session.get(api, timeout=60)
        r.raise_for_status()
        for e in r.json():
            if e.get("path") == self.filename:
                return int(e["size"]), (e.get("lfs") or {}).get("oid")
        raise FileNotFoundError(f"{self.filename} not in {DATASET}@{self.revision}")

    def cdn_url(self) -> str:
        # Follow the 302 once; the signed CDN URL is reused until it expires.
        r = self.session.get(self.resolve_url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
        r.raise_for_status()
        return r.url


def make_session() -> requests.Session:
    s = requests.Session()
    tok = os.environ.get("HF_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    s.headers["User-Agent"] = "xvi-extract/1.0"
    return s


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTPS using Range requests. Safe for one thread."""

    def __init__(self, src: HFSource):
        self.src = src
        self.session = make_session()
        self.url = src.cdn_url()
        self.size = src.size
        self.pos = 0
        self.bytes_read = 0
        self.requests_made = 0

    def readable(self): return True
    def seekable(self): return True
    def writable(self): return False
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        hdr = {"Range": f"bytes={self.pos}-{end}"}
        last = None
        for attempt in range(8):
            try:
                r = self.session.get(self.url, headers=hdr, timeout=180)
                self.requests_made += 1
                if r.status_code == 206:
                    data = r.content
                    self.pos += len(data)
                    self.bytes_read += len(data)
                    return data
                if r.status_code in (401, 403):      # signed CDN URL expired
                    self.url = self.src.cdn_url()
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:   # noqa: PERF203
                last = repr(e)
            time.sleep(min(30, 1.5 ** attempt))
        raise IOError(f"range read failed after retries: {last}")

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def read_footer(src: HFSource) -> pq.FileMetaData:
    f = RangeFile(src)
    return pq.ParquetFile(f).metadata


_local = threading.local()


def thread_parquet(src: HFSource, md: pq.FileMetaData) -> tuple[pq.ParquetFile, RangeFile]:
    key = (src.revision, src.filename)
    cache = getattr(_local, "pf", None)
    if cache is None or cache[0] != key:
        f = RangeFile(src)
        # pre_buffer coalesces the column-chunk reads of a row group into few large Range requests
        pf = pq.ParquetFile(f, metadata=md, pre_buffer=True)
        _local.pf = (key, pf, f)
    return _local.pf[1], _local.pf[2]


# --------------------------------------------------------------------------- #
# index: market_id -> row groups
# --------------------------------------------------------------------------- #
def cmd_index(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = HFSource(args.revision, TRADES_FILE)
    md = read_footer(src)
    names = [md.schema.column(j).name for j in range(md.num_columns)]
    mid_col = names.index("market_id")
    n = md.num_row_groups
    print(f"{TRADES_FILE}@{args.revision[:8]}: {md.num_rows:,} rows, {n} row groups, {src.size/1e9:.1f} GB, lfs sha256 {src.lfs_sha256}")
    idx_bytes = sum(md.row_group(i).column(mid_col).total_compressed_size for i in range(n))
    print(f"index scan will read ~{idx_bytes/1e6:.0f} MB of market_id column chunks with {args.workers} workers")

    # per-row-group stats (row groups are time-sorted; verify, since extract relies on nothing else)
    rg_rows = []
    for i in range(n):
        rg = md.row_group(i)
        st = rg.column(names.index("timestamp")).statistics
        comp = sum(rg.column(j).total_compressed_size for j in range(md.num_columns))
        rg_rows.append((i, rg.num_rows, int(st.min), int(st.max), comp))
    sorted_ok = all(rg_rows[i][3] <= rg_rows[i + 1][2] for i in range(n - 1))

    t0 = time.time(); done = [0]; lock = threading.Lock()
    results: dict[int, pa.Table] = {}

    def work(i):
        pf, f = thread_parquet(src, md)
        col = pf.read_row_group(i, columns=["market_id"])["market_id"]
        vc = pc.value_counts(col)
        t = pa.table({
            "market_id": vc.field("values"),
            "row_group": pa.array([i] * len(vc), pa.int32()),
            "n_rows": vc.field("counts"),
        })
        with lock:
            results[i] = t; done[0] += 1
            if done[0] % 50 == 0 or done[0] == n:
                print(f"  {done[0]}/{n} row groups indexed, {time.time()-t0:.0f}s", flush=True)
        return i

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, range(n)))

    index = pa.concat_tables([results[i] for i in range(n)]).sort_by([("market_id", "ascending"), ("row_group", "ascending")])
    pq.write_table(index, out / "market_rowgroups.parquet", compression="zstd")
    rg_tbl = pa.table({
        "row_group": pa.array([r[0] for r in rg_rows], pa.int32()),
        "num_rows": pa.array([r[1] for r in rg_rows], pa.int64()),
        "ts_min": pa.array([r[2] for r in rg_rows], pa.int64()),
        "ts_max": pa.array([r[3] for r in rg_rows], pa.int64()),
        "compressed_bytes": pa.array([r[4] for r in rg_rows], pa.int64()),
    })
    pq.write_table(rg_tbl, out / "rowgroup_stats.parquet")
    md.write_metadata_file(str(out / "trades.footer.parquet"))
    meta = {
        "dataset": DATASET, "revision": args.revision, "file": TRADES_FILE,
        "source": src.resolve_url, "source_size_bytes": src.size, "source_lfs_sha256": src.lfs_sha256,
        "num_rows": md.num_rows, "num_row_groups": n, "row_groups_time_sorted": sorted_ok,
        "distinct_markets": pc.count_distinct(index["market_id"]).as_py(),
        "index_pairs": index.num_rows, "built_at": utc_now(), "seconds": round(time.time() - t0, 1),
        "schema": [f"{md.schema.column(j).name}:{md.schema.column(j).physical_type}" for j in range(md.num_columns)],
    }
    (out / "index_manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"index written to {out}: {index.num_rows:,} (market, row_group) pairs, {meta['distinct_markets']:,} markets, {meta['seconds']}s")


# --------------------------------------------------------------------------- #
# lookup: how much would an extract cost
# --------------------------------------------------------------------------- #
def load_index(index_dir: Path):
    meta = json.loads((index_dir / "index_manifest.json").read_text())
    idx = pq.read_table(index_dir / "market_rowgroups.parquet")
    rgs = pq.read_table(index_dir / "rowgroup_stats.parquet").to_pandas().set_index("row_group")
    return meta, idx, rgs


def plan(index_dir: Path, market_ids: list[str]):
    meta, idx, rgs = load_index(index_dir)
    sel = idx.filter(pc.is_in(idx["market_id"], value_set=pa.array(market_ids))).to_pandas()
    per_market = {}
    for m in market_ids:
        s = sel[sel.market_id == m]
        per_market[m] = {"expected_rows": int(s.n_rows.sum()), "row_groups": sorted(s.row_group.tolist())}
    all_rgs = sorted(set(sel.row_group.tolist()))
    total_bytes = int(rgs.loc[all_rgs, "compressed_bytes"].sum()) if all_rgs else 0
    return meta, per_market, all_rgs, total_bytes


def cmd_lookup(args):
    meta, per_market, all_rgs, total_bytes = plan(Path(args.index), args.markets)
    _, _, rgs = load_index(Path(args.index))
    for m, p in per_market.items():
        if not p["row_groups"]:
            print(f"{m}: not present in trades.parquet@{meta['revision'][:8]}")
            continue
        lo, hi = p["row_groups"][0], p["row_groups"][-1]
        b = int(rgs.loc[p["row_groups"], "compressed_bytes"].sum())
        print(f"{m}: {p['expected_rows']:,} rows in {len(p['row_groups'])} row groups "
              f"[{lo}..{hi}], {dt.datetime.fromtimestamp(rgs.loc[lo,'ts_min'], dt.timezone.utc):%Y-%m-%d} to "
              f"{dt.datetime.fromtimestamp(rgs.loc[hi,'ts_max'], dt.timezone.utc):%Y-%m-%d}, ~{b/1e9:.2f} GB to fetch")
    print(f"union: {len(all_rgs)} row groups, ~{total_bytes/1e9:.2f} GB")


# --------------------------------------------------------------------------- #
# verify an existing extract directory
# --------------------------------------------------------------------------- #
def verify_extract(d: Path, expect_revision: str | None) -> dict:
    man = json.loads((d / "extraction_manifest.json").read_text())
    tp = d / "trades.parquet"
    actual = sha256_file(tp)
    recorded = man.get("sha256_trades_parquet")
    rep = {
        "dir": str(d), "revision": man.get("revision"), "market_ids": man.get("market_ids"),
        "sha256_recorded": recorded, "sha256_actual": actual, "checksum_ok": recorded == actual,
        "revision_ok": (expect_revision is None) or (man.get("revision") == expect_revision),
        "rows": pq.read_metadata(tp).num_rows,
    }
    return rep


def cmd_verify(args):
    rep = verify_extract(Path(args.dir), args.revision)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if rep["checksum_ok"] and rep["revision_ok"] else 1)


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def fetch_markets_metadata(revision: str, market_ids: list[str], index_dir: Path) -> tuple[pa.Table, dict]:
    """markets.parquet is 294 MB; read the needed columns once, cache them next to the index, filter locally."""
    cache, side = index_dir / "markets_meta.parquet", index_dir / "markets_meta.json"
    if cache.exists() and side.exists() and json.loads(side.read_text()).get("revision") == revision:
        tbl = pq.read_table(cache)
        prov = json.loads(side.read_text()); prov["markets_bytes_read"] = 0
    else:
        src = HFSource(revision, MARKETS_FILE)
        f = RangeFile(src)
        pf = pq.ParquetFile(f, pre_buffer=True)
        cols = [c for c in MARKET_COLS if c in pf.schema_arrow.names]
        tbl = pf.read(columns=cols)
        pq.write_table(tbl, cache, compression="zstd")
        prov = {"revision": revision, "markets_source": src.resolve_url, "markets_lfs_sha256": src.lfs_sha256,
                "markets_bytes_read": f.bytes_read, "markets_rows_total": tbl.num_rows, "cached_at": utc_now()}
        side.write_text(json.dumps(prov, indent=2))
    sel = tbl.filter(pc.is_in(tbl["id"], value_set=pa.array(market_ids)))
    return sel, prov


def cmd_extract(args):
    index_dir, out = Path(args.index), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    market_ids = [str(m) for m in args.markets]
    meta, per_market, all_rgs, total_bytes = plan(index_dir, market_ids)
    revision = meta["revision"]
    if args.revision and args.revision != revision:
        sys.exit(f"index was built at {revision}, requested {args.revision}; rebuild the index")
    started = utc_now(); t0 = time.time()
    missing = [m for m, p in per_market.items() if not p["row_groups"]]

    # 1. reuse rows from an existing, verified extract (e.g. news_attr/data/selected)
    reused: dict[str, pa.Table] = {}
    reuse_report = None
    if args.existing:
        ex = Path(args.existing)
        reuse_report = verify_extract(ex, revision)
        if reuse_report["checksum_ok"] and reuse_report["revision_ok"]:
            t = pq.read_table(ex / "trades.parquet")
            for m in market_ids:
                if m in (reuse_report["market_ids"] or []):
                    part = t.filter(pc.equal(t["market_id"], m))
                    if part.num_rows == per_market[m]["expected_rows"]:
                        reused[m] = part
                    else:
                        reuse_report.setdefault("count_mismatch", {})[m] = [part.num_rows, per_market[m]["expected_rows"]]
            print(f"reusing {list(reused)} from {ex} (checksum and revision verified)")
        else:
            print(f"NOT reusing {ex}: checksum_ok={reuse_report['checksum_ok']} revision_ok={reuse_report['revision_ok']}")

    to_fetch = [m for m in market_ids if m not in reused and m not in missing]
    fetch_set = set(to_fetch)
    rg_list = sorted({rg for m in to_fetch for rg in per_market[m]["row_groups"]})
    _, _, rgstats = load_index(index_dir)
    fetch_bytes = int(rgstats.loc[rg_list, "compressed_bytes"].sum()) if rg_list else 0
    print(f"fetching {len(rg_list)} row groups (~{fetch_bytes/1e9:.2f} GB) for markets {to_fetch}")

    # 2. fetch the row groups that contain the wanted markets
    src = HFSource(revision, TRADES_FILE)
    md = pq.read_metadata(index_dir / "trades.footer.parquet")
    if src.lfs_sha256 != meta["source_lfs_sha256"]:
        sys.exit("trades.parquet on the hub no longer matches the indexed file; rebuild the index")
    parts: dict[int, pa.Table] = {}; lock = threading.Lock(); done = [0]; bytes_read = [0]; reqs = [0]

    def work(i):
        pf, f = thread_parquet(src, md)
        b0, r0 = f.bytes_read, f.requests_made
        tbl = pf.read_row_group(i)
        keep = tbl.filter(pc.is_in(tbl["market_id"], value_set=pa.array(sorted(fetch_set))))
        with lock:
            parts[i] = keep; done[0] += 1
            bytes_read[0] += f.bytes_read - b0; reqs[0] += f.requests_made - r0
            if done[0] % 10 == 0 or done[0] == len(rg_list):
                el = time.time() - t0
                print(f"  {done[0]}/{len(rg_list)} row groups, {bytes_read[0]/1e9:.2f} GB, {el:.0f}s "
                      f"({bytes_read[0]/1e6/max(el,1):.1f} MB/s)", flush=True)

    if rg_list:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, rg_list))

    fetched = pa.concat_tables([parts[i] for i in rg_list]) if rg_list else None
    pieces = [p for p in reused.values()] + ([fetched] if fetched is not None else [])
    if not pieces:
        sys.exit("nothing to write")
    trades = pa.concat_tables(pieces, promote_options="default").sort_by(SORT_KEYS)

    # 3. verify row counts against the index, per market
    counts = trades.group_by("market_id").aggregate([("market_id", "count")]).to_pandas().set_index("market_id")["market_id_count"]
    summary, count_ok = [], True
    for m in market_ids:
        got = int(counts.get(m, 0)); exp = per_market[m]["expected_rows"]
        count_ok &= (got == exp)
        sub = trades.filter(pc.equal(trades["market_id"], m))
        summary.append({
            "market_id": m, "raw_fills": got, "expected_rows_from_index": exp, "row_count_ok": got == exp,
            "raw_usd_amount": float(pc.sum(sub["usd_amount"]).as_py() or 0) if got else 0.0,
            "first_timestamp": int(pc.min(sub["timestamp"]).as_py()) if got else None,
            "last_timestamp": int(pc.max(sub["timestamp"]).as_py()) if got else None,
            "distinct_transactions": pc.count_distinct(sub["transaction_hash"]).as_py() if got else 0,
            "distinct_wallets": pc.count_distinct(pa.concat_arrays([sub["maker"].combine_chunks(), sub["taker"].combine_chunks()])).as_py() if got else 0,
            "contracts": sorted(set(sub["contract"].to_pylist())) if got else [],
            "source": "reused_existing_extract" if m in reused else ("fetched" if m in fetch_set else "absent_at_revision"),
        })

    # 4. market metadata and token consistency
    mk, mprov = fetch_markets_metadata(revision, market_ids, index_dir)
    mk_pd = mk.to_pandas()
    tok = {r["id"]: (r["token1"], r["token2"]) for r in mk.to_pylist()}
    tr = trades.select(["market_id", "nonusdc_side", "asset_id"]).to_pandas()
    tr["expected_asset"] = [tok.get(m, (None, None))[0 if s == "token1" else 1] for m, s in zip(tr.market_id, tr.nonusdc_side)]
    asset_mismatch = int((tr.expected_asset != tr.asset_id).sum())
    for c in ("created_at", "end_date", "updated_at"):
        if c in mk_pd: mk_pd[c] = mk_pd[c].astype(str)
    mk_pd.to_csv(out / "markets.csv", index=False)
    (out / "markets.json").write_text(json.dumps(mk_pd.to_dict(orient="records"), indent=2))

    # 5. write outputs and manifest
    pq.write_table(trades, out / "trades.parquet", compression="zstd")
    trades.to_pandas().to_csv(out / "trades.csv", index=False) if args.csv else None
    manifest = {
        "status": "complete" if count_ok and not missing else "incomplete",
        "dataset": DATASET, "revision": revision, "market_ids": market_ids,
        "markets_absent_at_revision": missing,
        "source": src.resolve_url, "source_lfs_sha256": src.lfs_sha256, "source_size_bytes": src.size,
        **mprov,
        "method": "row-group index over market_id column + HTTP Range reads of matching row groups (no full download)",
        "index_manifest": meta,
        "row_groups_fetched": rg_list, "bytes_fetched_trades": bytes_read[0], "range_requests": reqs[0],
        "existing_extract": reuse_report,
        "started_at": started, "completed_at": utc_now(), "seconds": round(time.time() - t0, 1),
        "summary": summary, "all_row_counts_match_index": count_ok,
        "asset_id_matches_market_token": asset_mismatch == 0, "asset_id_mismatch_rows": asset_mismatch,
        "markets_metadata_found": sorted(mk_pd["id"].tolist()),
        "note_on_outcome_prices": "markets.parquet outcome_prices is Gamma's final price string, not the on-chain payout vector; verify settlement separately before scoring.",
        "sort_order": [k for k, _ in SORT_KEYS],
        "sha256_trades_parquet": sha256_file(out / "trades.parquet"),
        "schema": trades.schema.to_string(show_schema_metadata=False).split("\n"),
    }
    (out / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({k: manifest[k] for k in ("status", "revision", "row_groups_fetched", "bytes_fetched_trades", "seconds", "all_row_counts_match_index", "asset_id_mismatch_rows")}, default=str))
    for s in summary:
        print(f"  {s['market_id']}: {s['raw_fills']:,} fills ({s['source']}), count_ok={s['row_count_ok']}, "
              f"{s['distinct_wallets']:,} wallets, contracts={s['contracts']}")
    print(f"wrote {out/'trades.parquet'} ({trades.num_rows:,} rows), markets.csv, markets.json, extraction_manifest.json")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("index"); a.add_argument("--out", default="data/index"); a.add_argument("--revision", default=DEFAULT_REVISION); a.add_argument("--workers", type=int, default=8); a.set_defaults(fn=cmd_index)
    a = sp.add_parser("lookup"); a.add_argument("--markets", nargs="+", required=True); a.add_argument("--index", default="data/index"); a.set_defaults(fn=cmd_lookup)
    a = sp.add_parser("extract"); a.add_argument("--markets", nargs="+", required=True); a.add_argument("--index", default="data/index"); a.add_argument("--out", default="data/selected")
    a.add_argument("--existing", help="an earlier extract dir to reuse rows from after checksum+revision verification"); a.add_argument("--revision", default=None)
    a.add_argument("--workers", type=int, default=6); a.add_argument("--csv", action="store_true"); a.set_defaults(fn=cmd_extract)
    _add_fills_parser(sp)
    a = sp.add_parser("verify"); a.add_argument("--dir", required=True); a.add_argument("--revision", default=DEFAULT_REVISION); a.set_defaults(fn=cmd_verify)
    args = p.parse_args(); args.fn(args)



# --------------------------------------------------------------------------- #
# fills: raw OrderFilled events (maker AND taker orders) for the selected markets
# --------------------------------------------------------------------------- #
ORDERFILLED_FILE = "orderfilled.parquet"
FILL_COLS = ["timestamp", "block_number", "log_index", "contract", "order_hash", "maker", "taker", "maker_asset_id",
             "taker_asset_id", "maker_amount_filled", "taker_amount_filled", "maker_fee", "taker_fee", "protocol_fee"]
KNOWN_EXCHANGES = {  # Polygon addresses that appear as `taker` on an order's own OrderFilled event
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e": "CTF_EXCHANGE",
    "0xc5d563a36ae78145c45a50134d48a1215220f80a": "NEGRISK_CTF_EXCHANGE",
}


def _u256_le(col: pa.ChunkedArray) -> pa.Array:
    # amounts are uint256 stored as 32-byte little-endian; values in this dataset fit int64 (micro-units)
    import numpy as np
    buf = np.frombuffer(b"".join(col.combine_chunks().to_pylist()), dtype="<u8").reshape(-1, 4)
    assert (buf[:, 1:] == 0).all(), "uint256 value exceeds 64 bits"
    return pa.array(buf[:, 0].astype("int64"))


def cmd_fills(args):
    """Read every OrderFilled row whose non-USDC asset is one of the selected markets' tokens.

    trades.parquet keeps only the maker-order events and derives the taker's side as the mirror of the
    maker's, which is wrong for MINT/MERGE matches. The taker order's own event (taker = exchange
    contract) carries the taker's real side, token, amounts and fee. orderfilled.parquet is partitioned by
    month with unsorted row groups inside each month, so every row group of each active month is read,
    with column projection. ``order_hash`` is retained so later analysis can recognise partial executions
    of the same order; transaction_hash is skipped because (block_number, log_index) is the join key to
    trades.parquet.
    """
    import datetime as dt
    sel, index_dir = Path(args.selected), Path(args.index)
    cache = sel / "cache_orderfilled"; cache.mkdir(exist_ok=True)
    man = json.loads((sel / "extraction_manifest.json").read_text())
    revision = man["revision"]
    markets = json.loads((sel / "markets.json").read_text())
    tokens = {m["token1"]: (m["id"], "token1") for m in markets} | {m["token2"]: (m["id"], "token2") for m in markets}
    trades = pq.read_table(sel / "trades.parquet", columns=["market_id", "timestamp", "block_number", "log_index"])
    # months to scan, from the selected trades' own time span
    months = set()
    for m in markets:
        sub = trades.filter(pc.equal(trades["market_id"], m["id"]))
        if sub.num_rows == 0: continue
        a = dt.datetime.fromtimestamp(pc.min(sub["timestamp"]).as_py(), dt.timezone.utc)
        b = dt.datetime.fromtimestamp(pc.max(sub["timestamp"]).as_py(), dt.timezone.utc)
        cur = dt.datetime(a.year, a.month, 1, tzinfo=dt.timezone.utc)
        while cur <= b:
            months.add(f"{cur.year}-{cur.month:02d}"); cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    months = sorted(months)

    src = HFSource(revision, ORDERFILLED_FILE)
    footer = index_dir / "orderfilled.footer.parquet"
    if footer.exists():
        md = pq.read_metadata(footer)
    else:
        md = read_footer(src); md.write_metadata_file(str(footer))
    names = [md.schema.column(j).name for j in range(md.num_columns)]
    tsi = names.index("timestamp"); need = [names.index(c) for c in FILL_COLS]
    by_month: dict[str, list[int]] = {}
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(tsi).statistics
        ym = f"{dt.datetime.fromtimestamp(st.min, dt.timezone.utc):%Y-%m}"
        ym2 = f"{dt.datetime.fromtimestamp(st.max, dt.timezone.utc):%Y-%m}"
        if ym != ym2: sys.exit(f"row group {i} spans months {ym}..{ym2}; month partition assumption broken")
        by_month.setdefault(ym, []).append(i)
    # A cache produced by an older version can be read by scoring, but it must be refreshed here so a new
    # extraction does not silently omit the newly retained order_hash column.
    todo = [ym for ym in months if not (cache / f"{ym}.parquet").exists()
            or "order_hash" not in pq.read_schema(cache / f"{ym}.parquet").names]
    est = sum(md.row_group(i).column(j).total_compressed_size for ym in todo for i in by_month.get(ym, []) for j in need)
    print(f"months {months}; {len(todo)} to fetch, ~{est/1e9:.2f} GB projected over {sum(len(by_month.get(ym, [])) for ym in todo)} row groups")
    token_arr = pa.array(list(tokens))
    t0 = time.time(); lock = threading.Lock(); stats = {"bytes": 0, "reqs": 0}

    def work(i):
        pf, f = thread_parquet(src, md)
        b0, r0 = f.bytes_read, f.requests_made
        tbl = pf.read_row_group(i, columns=FILL_COLS)
        keep = tbl.filter(pc.or_(pc.is_in(tbl["maker_asset_id"], value_set=token_arr), pc.is_in(tbl["taker_asset_id"], value_set=token_arr)))
        with lock:
            stats["bytes"] += f.bytes_read - b0; stats["reqs"] += f.requests_made - r0
        return i, keep

    for ym in todo:
        rgs = by_month.get(ym, [])
        parts = {}
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, keep in ex.map(work, rgs):
                parts[i] = keep
        tbl = pa.concat_tables([parts[i] for i in rgs]) if rgs else pa.table({c: [] for c in FILL_COLS})
        if tbl.num_rows:
            cols = {c: tbl[c] for c in FILL_COLS if c not in ("maker_amount_filled", "taker_amount_filled", "maker_fee", "taker_fee", "protocol_fee")}
            for c in ("maker_amount_filled", "taker_amount_filled", "maker_fee", "taker_fee", "protocol_fee"):
                cols[c] = _u256_le(tbl[c])
            tbl = pa.table(cols)
        pq.write_table(tbl, cache / f"{ym}.parquet", compression="zstd")
        el = time.time() - t0
        print(f"  {ym}: {len(rgs)} row groups, {tbl.num_rows:,} matching rows, cumulative {stats['bytes']/1e9:.2f} GB in {el:.0f}s ({stats['bytes']/1e6/max(el,1):.0f} MB/s)", flush=True)

    fills = pa.concat_tables([pq.read_table(cache / f"{ym}.parquet") for ym in months]).sort_by(SORT_KEYS)
    # classify rows: the order's own event has taker == exchange; on maker events taker == the taker's wallet
    makers = set(pc.unique(fills["maker"]).to_pylist())
    takers_seen = set(pc.unique(fills["taker"]).to_pylist())
    exchange_addrs = {a for a in takers_seen if a not in makers}
    unknown_ex = {a for a in exchange_addrs if a.lower() not in KNOWN_EXCHANGES}
    role = pa.array(["taker_order" if a in exchange_addrs else "maker_order" for a in fills["taker"].to_pylist()])
    nonusdc = pc.if_else(pc.equal(fills["maker_asset_id"], "0"), fills["taker_asset_id"], fills["maker_asset_id"])
    mid = pa.array([tokens.get(a, (None, None))[0] for a in nonusdc.to_pylist()])
    side = pa.array([tokens.get(a, (None, None))[1] for a in nonusdc.to_pylist()])
    fills = fills.append_column("event_role", role).append_column("asset_id", nonusdc).append_column("market_id", mid).append_column("nonusdc_side", side)
    # join check against trades.parquet on (block_number, log_index)
    key_tr = set(zip(trades["block_number"].to_pylist(), trades["log_index"].to_pylist()))
    key_of = list(zip(fills["block_number"].to_pylist(), fills["log_index"].to_pylist()))
    in_trades = pa.array([k in key_tr for k in key_of])
    fills = fills.append_column("in_trades_parquet", in_trades)
    n_maker = pc.sum(pc.equal(role, "maker_order")).as_py(); n_taker = fills.num_rows - n_maker
    maker_in_trades = pc.sum(pc.and_(pc.equal(role, "maker_order"), in_trades)).as_py()
    trades_in_fills = len(key_tr & set(key_of))
    pq.write_table(fills, sel / "orderfilled_selected.parquet", compression="zstd")
    rep = {
        "dataset": DATASET, "revision": revision, "file": ORDERFILLED_FILE, "source": src.resolve_url,
        "source_lfs_sha256": src.lfs_sha256, "months_scanned": months, "columns": FILL_COLS,
        "bytes_fetched_this_run": stats["bytes"], "range_requests_this_run": stats["reqs"], "seconds": round(time.time() - t0, 1),
        "rows": fills.num_rows, "maker_order_rows": n_maker, "taker_order_rows": n_taker,
        "maker_rows_present_in_trades_parquet": maker_in_trades, "trades_parquet_rows_found_in_orderfilled": trades_in_fills,
        "trades_parquet_rows_total": trades.num_rows,
        "exchange_addresses_detected": sorted(exchange_addrs), "exchange_addresses_unrecognised": sorted(unknown_ex),
        "amount_encoding": "uint256 little-endian, decoded to int64 micro-units (1e6 = 1 USDC or 1 share)",
        "order_hash_caveat": "order_hash identifies repeated executions of the same submitted order, but OrderFilled data contains no unfilled or cancelled orders and a maker order's first observed fill is only an upper bound on its placement time",
        "completed_at": utc_now(), "sha256_orderfilled_selected_parquet": sha256_file(sel / "orderfilled_selected.parquet"),
    }
    (sel / "fills_manifest.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: rep[k] for k in ("rows", "maker_order_rows", "taker_order_rows", "maker_rows_present_in_trades_parquet",
                                          "trades_parquet_rows_found_in_orderfilled", "trades_parquet_rows_total", "exchange_addresses_detected", "exchange_addresses_unrecognised")}))


def _add_fills_parser(sp):
    a = sp.add_parser("fills", help="pull raw OrderFilled events (maker and taker orders) for the selected markets from orderfilled.parquet")
    a.add_argument("--selected", default="data/selected"); a.add_argument("--index", default="data/index")
    a.add_argument("--workers", type=int, default=8); a.set_defaults(fn=cmd_fills)


if __name__ == "__main__":
    main()
