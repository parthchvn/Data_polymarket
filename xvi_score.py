#!/usr/bin/env python3
"""
xvi_score.py - actor decisions and settlement payoffs for selected Polymarket markets.

Pipeline (each step reads cached files; nothing rescans the archive):

  data/selected/orderfilled_selected.parquet   raw OrderFilled events, maker AND taker orders   (xvi_extract.py fills)
  data/selected/trades.parquet                 Wang's processed maker rows, used for linkage checks (xvi_extract.py extract)
  data/settlement/<condition_id>.json          on-chain payout vector + resolution timestamp     (xvi_settlement.py)
        |
        v
  data/scored/market_decisions.parquet         one row per actor decision, with decision_payoff_usd
  data/scored/actor_summary.parquet            per (wallet, market) sums and coverage
  data/scored/actor_summary_all_markets.parquet
  data/scored/market_decisions.jsonl.gz        training export
  data/scored/scoring_manifest.json            counts, checks, evidence

Commands:
  python xvi_score.py run      [--selected data/selected] [--settlement data/settlement] [--out data/scored]
  python xvi_score.py wallet   --wallet 0x... [--market 507300] [--out data/scored]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MICRO = 1_000_000


def actor_id_of(wallet: str) -> int:
    """Stable integer actor id derived from the wallet address (same wallet -> same id across markets and reruns)."""
    return int(hashlib.sha256(wallet.lower().encode()).hexdigest()[:12], 16)
EPS_SHARES = 1e-6
V1_CONTRACTS = {"CTF_EXCHANGE", "NEGRISK_CTF_EXCHANGE"}
V2_CONTRACTS = {"CTF_EXCHANGE_V2"}
DECISION_COLS = [
    "decision_id", "source_execution_id", "market_id", "condition_id", "timestamp", "datetime_utc", "actor_id", "actor_wallet", "actor_role",
    "actor_seq", "actor_market_seq", "actor_n_decisions", "prev_decision_timestamp",
    "counterparty_wallet", "action", "outcome_token", "outcome_label", "asset_id", "shares", "execution_price", "cash_amount_usd",
    "fee_amount", "fee_asset", "transaction_hash", "order_hash", "block_number", "log_index", "contract", "order_group_id", "source_revision",
    "action_semantics", "action_verified", "fill_group_consistent", "in_trades_parquet",
    "settlement_payout", "settlement_verified", "settlement_source", "settlement_timestamp", "payout_verified_at", "label_available_at",
    "decision_payoff_usd", "decision_payoff_after_fee_usd", "score_status", "score_note", "label_public_at_decision",
]


# --------------------------------------------------------------------------- #
# 1. normalize raw fills into actor decisions
# --------------------------------------------------------------------------- #
def normalize_fills(fills: pd.DataFrame, markets: list[dict], trades: pd.DataFrame | None, source_revision: str) -> tuple[pd.DataFrame, dict]:
    """One decision per OrderFilled event, attributed to the order's owner (the event's `maker` field).

    ``order_hash`` is carried through when the extract provides it. Older extracts do not have the column,
    so their decisions receive a null value. Repeated fills that share an order hash remain separate
    execution decisions; this function never collapses them into one row.

    Exchange semantics used:
      V1 contracts (CTF_EXCHANGE, NEGRISK_CTF_EXCHANGE): exactly one leg is collateral (asset id "0").
          maker_asset_id == "0"  -> the owner BUYs taker_asset_id: pays maker_amount_filled, receives taker_amount_filled shares
          taker_asset_id == "0"  -> the owner SELLs maker_asset_id: gives maker_amount_filled shares, receives taker_amount_filled
      V2 contract (CTF_EXCHANGE_V2): first id field is a side flag ("0" BUY, "1" SELL), second is always the token id.
          BUY  -> pays maker_amount_filled, receives taker_amount_filled shares
          SELL -> gives maker_amount_filled shares, receives taker_amount_filled
      The order's own event has taker == exchange contract (event_role == taker_order); on a maker order's event the
      `taker` column is the taker's wallet, but that row says nothing reliable about the taker's side (MINT/MERGE).
    """
    tok = {}
    for m in markets:
        tok[m["token1"]] = (m["id"], m["condition_id"], "token1", m["answer1"])
        tok[m["token2"]] = (m["id"], m["condition_id"], "token2", m["answer2"])
    f = fills.copy()
    n_raw = len(f)
    f = f.drop_duplicates(["block_number", "log_index"])
    n_dup = n_raw - len(f)

    is_v2 = f["contract"].isin(V2_CONTRACTS)
    is_v1 = f["contract"].isin(V1_CONTRACTS)
    m0 = f["maker_asset_id"].eq("0"); t0 = f["taker_asset_id"].eq("0"); m1 = f["maker_asset_id"].eq("1")
    action = pd.Series(pd.NA, index=f.index, dtype="object"); token_id = pd.Series(pd.NA, index=f.index, dtype="object")
    cash = pd.Series(np.nan, index=f.index); shares = pd.Series(np.nan, index=f.index); sem = pd.Series("unknown", index=f.index, dtype="object")
    # V1
    sel = is_v1 & m0 & ~t0
    action[sel], token_id[sel], cash[sel], shares[sel], sem[sel] = "BUY", f.loc[sel, "taker_asset_id"], f.loc[sel, "maker_amount_filled"], f.loc[sel, "taker_amount_filled"], "v1_collateral_leg"
    sel = is_v1 & t0 & ~m0
    action[sel], token_id[sel], cash[sel], shares[sel], sem[sel] = "SELL", f.loc[sel, "maker_asset_id"], f.loc[sel, "taker_amount_filled"], f.loc[sel, "maker_amount_filled"], "v1_collateral_leg"
    # V2
    sel = is_v2 & m0
    action[sel], token_id[sel], cash[sel], shares[sel], sem[sel] = "BUY", f.loc[sel, "taker_asset_id"], f.loc[sel, "maker_amount_filled"], f.loc[sel, "taker_amount_filled"], "v2_side_flag"
    sel = is_v2 & m1
    action[sel], token_id[sel], cash[sel], shares[sel], sem[sel] = "SELL", f.loc[sel, "taker_asset_id"], f.loc[sel, "taker_amount_filled"], f.loc[sel, "maker_amount_filled"], "v2_side_flag"

    # rows whose semantics could not be decoded keep their market attribution (the fills step tagged the traded asset)
    if "asset_id" in f:
        token_id = token_id.where(token_id.notna(), f["asset_id"].where(f["asset_id"].isin(tok)))
    meta = token_id.map(lambda t: tok.get(t) if isinstance(t, str) else None)
    d = pd.DataFrame({
        "block_number": f["block_number"].astype("int64"), "log_index": f["log_index"].astype("int64"),
        "timestamp": f["timestamp"].astype("int64"), "contract": f["contract"],
        "actor_wallet": f["maker"], "actor_role": np.where(f["event_role"].eq("taker_order"), "taker", "maker"),
        "counterparty_wallet": np.where(f["event_role"].eq("taker_order"), None, f["taker"]),
        "action": action, "asset_id": token_id,
        "market_id": meta.map(lambda x: x[0] if x else None), "condition_id": meta.map(lambda x: x[1] if x else None),
        "outcome_token": meta.map(lambda x: x[2] if x else None), "outcome_label": meta.map(lambda x: x[3] if x else None),
        "shares_micro": shares, "cash_micro": cash, "fee_micro": f["maker_fee"].astype("int64"),
        "action_semantics": sem, "in_trades_parquet": f["in_trades_parquet"].astype(bool),
    }, index=f.index)
    d["action_verified"] = d["action"].notna() & d["market_id"].notna() & (d["shares_micro"] > 0)
    d["shares"] = d["shares_micro"] / MICRO
    d["cash_amount_usd"] = d["cash_micro"] / MICRO
    d["execution_price"] = np.where(d["shares_micro"] > 0, d["cash_micro"] / d["shares_micro"].replace(0, np.nan), np.nan)
    d["fee_amount"] = d["fee_micro"] / MICRO
    # On CTF_EXCHANGE_V2 the fee is settled in collateral on both sides: a BUY pays cash + fee, a SELL receives cash - fee
    # (verified from Polygon receipts, see scoring_manifest.json). Only V2 events carry non-zero fees in this data.
    d["fee_asset"] = np.where(d["fee_micro"] > 0, "usd", None)
    d["datetime_utc"] = pd.to_datetime(d["timestamp"], unit="s", utc=True)
    d["decision_id"] = d["block_number"].astype(str) + "-" + d["log_index"].astype(str)
    d["source_execution_id"] = "orderfilled:" + d["decision_id"]
    d["source_revision"] = source_revision
    d["order_hash"] = f["order_hash"] if "order_hash" in f.columns else None
    # taker order group: a taker order's fills share (block, taker wallet); the taker's own event is the decision
    d["order_group_id"] = np.where(d["actor_role"].eq("taker"),
                                   d["block_number"].astype(str) + ":" + d["actor_wallet"],
                                   d["block_number"].astype(str) + ":" + d["counterparty_wallet"].astype(str))
    # fill-group consistency: sum of maker fill shares == taker order shares for the same (block, taker wallet)
    mk = d[d["actor_role"].eq("maker")].groupby("order_group_id")["shares_micro"].sum()
    tk = d[d["actor_role"].eq("taker")].groupby("order_group_id")["shares_micro"].sum()
    cmp = pd.concat([mk.rename("mk"), tk.rename("tk")], axis=1)
    consistent_groups = cmp.index[(cmp["mk"].notna()) & (cmp["tk"].notna()) & (cmp["mk"] == cmp["tk"])]
    d["fill_group_consistent"] = d["order_group_id"].isin(set(consistent_groups))
    groups_no_taker = int(cmp["tk"].isna().sum()); groups_no_maker = int(cmp["mk"].isna().sum()); groups_mismatch = int(len(cmp) - len(consistent_groups) - groups_no_taker - groups_no_maker)
    # transaction hash: taker rows inherit it from their maker fills (same transaction) when unambiguous
    d["transaction_hash"] = None
    if trades is not None and len(trades):
        key = trades.set_index(["block_number", "log_index"])["transaction_hash"]
        idx = pd.MultiIndex.from_arrays([d["block_number"], d["log_index"]])
        d["transaction_hash"] = key.reindex(idx).to_numpy()
        mk_tx = d[d["actor_role"].eq("maker") & d["transaction_hash"].notna()][["order_group_id", "log_index", "transaction_hash"]]
        need = d["actor_role"].eq("taker") & d["transaction_hash"].isna()
        if need.any() and len(mk_tx):
            # a taker event follows its maker fills within the same transaction: take the nearest preceding maker fill's hash
            t = d.loc[need, ["order_group_id", "log_index"]].reset_index()
            m = pd.merge_asof(t.sort_values("log_index"), mk_tx.sort_values("log_index"), on="log_index", by="order_group_id", direction="backward")
            d.loc[m["index"].to_numpy(), "transaction_hash"] = m["transaction_hash"].to_numpy()
    d = d.sort_values(["timestamp", "block_number", "log_index"]).reset_index(drop=True)
    report = {"raw_rows": n_raw, "duplicate_rows_removed": n_dup, "decisions": len(d),
              "maker_decisions": int(d["actor_role"].eq("maker").sum()), "taker_decisions": int(d["actor_role"].eq("taker").sum()),
              "action_unverified": int((~d["action_verified"]).sum()),
              "fill_groups": int(len(cmp)), "fill_groups_consistent": int(len(consistent_groups)),
              "fill_groups_missing_taker_event": groups_no_taker, "fill_groups_missing_maker_events": groups_no_maker, "fill_groups_share_mismatch": groups_mismatch,
              "taker_rows_without_transaction_hash": int((d["actor_role"].eq("taker") & d["transaction_hash"].isna()).sum()),
              "order_hash_available": "order_hash" in f.columns,
              "rows_without_order_hash": int(d["order_hash"].isna().sum()),
              "distinct_order_hashes": int(d["order_hash"].nunique(dropna=True))}
    return d, report


def validate_against_trades(d: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Wang's maker rows carry price/usd/token amounts rounded to 2 dp; check they agree with the raw events."""
    t = trades.set_index(["block_number", "log_index"])
    m = d[d["actor_role"].eq("maker")].set_index(["block_number", "log_index"])
    j = m.join(t[["price", "usd_amount", "token_amount", "maker_direction", "nonusdc_side"]], how="inner")
    return {
        "maker_decisions_matched_to_trades_parquet": int(len(j)),
        "maker_decisions_absent_from_trades_parquet": int(len(m) - len(j)),
        "direction_agrees": int((j["maker_direction"] == j["action"]).sum()), "token_agrees": int((j["nonusdc_side"] == j["outcome_token"]).sum()),
        "max_abs_usd_diff": float((j["usd_amount"] - j["cash_amount_usd"]).abs().max()),
        "max_abs_shares_diff": float((j["token_amount"] - j["shares"]).abs().max()),
        "rows_usd_diff_over_1c": int(((j["usd_amount"] - j["cash_amount_usd"]).abs() > 0.01).sum()),
        "rows_shares_diff_over_1c": int(((j["token_amount"] - j["shares"]).abs() > 0.01).sum()),
        "trades_parquet_price_times_shares_vs_usd_max_diff": float((trades["price"] * trades["token_amount"] - trades["usd_amount"]).abs().max()),
    }


# --------------------------------------------------------------------------- #
# 2. score each decision independently against verified settlement
# --------------------------------------------------------------------------- #
def score_decisions(d: pd.DataFrame, settlement: dict[str, dict]) -> pd.DataFrame:
    d = d.copy()
    for c in ("settlement_payout", "decision_payoff_usd", "decision_payoff_after_fee_usd"): d[c] = np.nan
    for c in ("settlement_verified",): d[c] = False
    for c in ("settlement_source", "settlement_timestamp", "payout_verified_at", "label_available_at", "score_status", "score_note"): d[c] = None
    d["settlement_timestamp"] = d["settlement_timestamp"].astype("object"); d["label_available_at"] = d["label_available_at"].astype("object")
    for mid, s in settlement.items():
        rows = d["market_id"].eq(mid)
        if not rows.any(): continue
        ok = bool(s.get("settlement_verified")) and s.get("payout_token1") is not None
        d.loc[rows, "settlement_source"] = s.get("settlement_source")
        d.loc[rows, "payout_verified_at"] = s.get("payout_verified_at")
        d.loc[rows, "settlement_verified"] = ok
        if s.get("settlement_timestamp") is not None:
            d.loc[rows, "settlement_timestamp"] = int(s["settlement_timestamp"]); d.loc[rows, "label_available_at"] = int(s["settlement_timestamp"])
        if not ok:
            d.loc[rows, "score_status"] = "unavailable_settlement_unverified"
            d.loc[rows, "score_note"] = f"settlement status: {s.get('status')}"
            continue
        v = np.where(d.loc[rows, "outcome_token"].eq("token1"), float(s["payout_token1"]), float(s["payout_token2"]))
        d.loc[rows, "settlement_payout"] = v
    # formula, exact in micro-units: BUY q*v - u ; SELL u - q*v
    q, u, v = d["shares_micro"], d["cash_micro"], d["settlement_payout"]
    scorable = d["action_verified"] & d["settlement_verified"]
    buy = d["action"].eq("BUY")
    payoff = np.where(buy, q * v - u, u - q * v) / MICRO
    fee = d["fee_micro"]
    net = (np.where(buy, q * v - u, u - q * v) - fee) / MICRO
    d.loc[scorable, "decision_payoff_usd"] = np.round(payoff[scorable], 6)
    d.loc[scorable, "decision_payoff_after_fee_usd"] = np.round(net[scorable], 6)
    d.loc[scorable, "score_status"] = "scored"
    d.loc[scorable, "score_note"] = "before fees; after-fee value subtracts the collateral fee charged to this order"
    bad = ~d["action_verified"]
    d.loc[bad, "score_status"] = "unavailable_action_unverified"
    d.loc[bad, "score_note"] = "could not establish side/token/amounts from the exchange event"
    d.loc[~scorable & d["score_status"].isna(), "score_status"] = "unavailable_settlement_unverified"
    # hindsight flag: the payoff was already public when this decision was executed
    st = pd.to_numeric(d["settlement_timestamp"], errors="coerce")
    d["label_public_at_decision"] = (st.notna() & (d["timestamp"] >= st)).astype(bool)
    return d


# --------------------------------------------------------------------------- #
# 3. aggregate by actor
# --------------------------------------------------------------------------- #

def add_actor_sequences(d: pd.DataFrame) -> pd.DataFrame:
    """actor_id plus the order in which each actor acted: actor_seq over all imported markets, actor_market_seq
    within one market. Ordering is execution time, then block, then log index; nothing sub-second is invented."""
    d = d.sort_values(["timestamp", "block_number", "log_index"], kind="stable").reset_index(drop=True)
    d["actor_id"] = d["actor_wallet"].map(actor_id_of).astype("int64")
    d["actor_seq"] = d.groupby("actor_id").cumcount() + 1
    d["actor_market_seq"] = d.groupby(["actor_id", "market_id"]).cumcount() + 1
    d["actor_n_decisions"] = d.groupby("actor_id")["actor_seq"].transform("max")
    d["prev_decision_timestamp"] = d.groupby("actor_id")["timestamp"].shift(1).astype("Int64")
    return d


def aggregate_actors(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["timestamp", "block_number", "log_index"])
    scored = d["score_status"].eq("scored")
    sign = np.where(d["action"].eq("BUY"), 1.0, -1.0) * d["shares"].fillna(0)
    d = d.assign(_signed=sign, _scored=scored,
                 _t1=np.where(d["outcome_token"].eq("token1"), sign, 0.0), _t2=np.where(d["outcome_token"].eq("token2"), sign, 0.0))
    d["_cum1"] = d.groupby(["actor_wallet", "market_id"])["_t1"].cumsum(); d["_cum2"] = d.groupby(["actor_wallet", "market_id"])["_t2"].cumsum()
    g = d.groupby(["actor_wallet", "market_id"])
    out = pd.DataFrame({
        "n_decisions": g.size(),
        "n_scored": g["_scored"].sum().astype(int),
        "n_buy": g["action"].apply(lambda s: int((s == "BUY").sum())),
        "n_sell": g["action"].apply(lambda s: int((s == "SELL").sum())),
        "n_as_taker": g["actor_role"].apply(lambda s: int((s == "taker").sum())),
        "n_as_maker": g["actor_role"].apply(lambda s: int((s == "maker").sum())),
        "actor_market_settlement_payoff_usd": g["decision_payoff_usd"].sum(min_count=1),
        "actor_market_settlement_payoff_after_fee_usd": g["decision_payoff_after_fee_usd"].sum(min_count=1),
        "fees_usd_total": g["fee_amount"].sum(),
        "cash_paid_usd": g.apply(lambda x: float(x.loc[x["action"].eq("BUY"), "cash_amount_usd"].sum())),
        "cash_received_usd": g.apply(lambda x: float(x.loc[x["action"].eq("SELL"), "cash_amount_usd"].sum())),
        "net_position_token1": g["_t1"].sum(), "net_position_token2": g["_t2"].sum(),
        "min_running_position_token1": g["_cum1"].min(), "min_running_position_token2": g["_cum2"].min(),
        "first_timestamp": g["timestamp"].min(), "last_timestamp": g["timestamp"].max(),
        "unscored_reasons": g["score_status"].apply(lambda s: ",".join(sorted(set(s[s != "scored"].dropna()))) or None),
    }).reset_index()
    out.insert(1, "actor_id", out["actor_wallet"].map(actor_id_of).astype("int64"))
    out["n_unscored"] = out["n_decisions"] - out["n_scored"]
    out["coverage_status"] = np.where(out["n_unscored"] == 0, "all_observed_decisions_scored", "partial: some decisions unscored")
    out["external_inventory_detected"] = (out["min_running_position_token1"] < -EPS_SHARES) | (out["min_running_position_token2"] < -EPS_SHARES)
    out["aggregate_label"] = "Total settlement payoff of observed trades (this market, before fees)"
    out["accounting_note"] = np.where(
        out["external_inventory_detected"],
        "sold more than bought on the exchange: holdings came from transfers, splits, merges or neg-risk conversions not in the fill history; not a verified final profit",
        "sum of independently scored exchange fills; transfers, splits, merges and conversions are not observed, so this is not certified as final trading profit")
    return out.sort_values("actor_market_settlement_payoff_usd", ascending=False).reset_index(drop=True)


def aggregate_all_markets(summary: pd.DataFrame, market_ids: list[str]) -> pd.DataFrame:
    g = summary.groupby("actor_wallet")
    out = pd.DataFrame({
        "markets_traded": g["market_id"].nunique(), "n_decisions": g["n_decisions"].sum(), "n_scored": g["n_scored"].sum(), "n_unscored": g["n_unscored"].sum(),
        "settlement_payoff_usd_imported_markets": g["actor_market_settlement_payoff_usd"].sum(min_count=1),
        "external_inventory_detected_any": g["external_inventory_detected"].any(),
    }).reset_index()
    out.insert(1, "actor_id", out["actor_wallet"].map(actor_id_of).astype("int64"))
    out["coverage_label"] = f"sum over imported markets {','.join(market_ids)} only; not lifetime Polymarket profit"
    return out.sort_values("settlement_payoff_usd_imported_markets", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. run
# --------------------------------------------------------------------------- #
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def cmd_run(args):
    sel, out = Path(args.selected), Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ext = json.loads((sel / "extraction_manifest.json").read_text())
    fills_man = json.loads((sel / "fills_manifest.json").read_text())
    markets = json.loads((sel / "markets.json").read_text())
    fills = pd.read_parquet(sel / "orderfilled_selected.parquet")
    trades = pd.read_parquet(sel / "trades.parquet")
    import xvi_settlement
    settlement = xvi_settlement.load_or_fetch(markets, Path(args.settlement))

    d, norm_rep = normalize_fills(fills, markets, trades, ext["revision"])
    val = validate_against_trades(d, trades)
    d = score_decisions(d, settlement)
    d = add_actor_sequences(d)
    summary = aggregate_actors(d)
    all_mk = aggregate_all_markets(summary, [m["id"] for m in markets])

    # consistency: per-market sums of decisions == sums in actor summary
    checks = {}
    for m in markets:
        mid = m["id"]; dm = d[d["market_id"].eq(mid)]; sm = summary[summary["market_id"].eq(mid)]
        checks[mid] = {
            "decisions": int(len(dm)), "scored": int(dm["score_status"].eq("scored").sum()),
            "unscored_by_reason": dm.loc[~dm["score_status"].eq("scored"), "score_status"].value_counts().to_dict(),
            "sum_decision_payoff_usd": round(float(dm["decision_payoff_usd"].sum()), 6),
            "sum_actor_summary_usd": round(float(sm["actor_market_settlement_payoff_usd"].sum()), 6),
            "actors": int(len(sm)),
            # zero-sum sanity: every fill has a buyer and a seller of complementary or identical claims, so before fees the
            # market-wide sum of settlement payoffs must be ~0 when every counterparty is observed (it is not exactly:
            # MINT/MERGE fills have their counterparty in the other token, still inside this market)
            "market_wide_payoff_sum_usd": round(float(dm["decision_payoff_usd"].sum()), 2),
            "maker_rows_absent_from_trades_parquet": int((dm["actor_role"].eq("maker") & ~dm["in_trades_parquet"]).sum()),
            "decisions_executed_at_or_after_settlement": int(dm["label_public_at_decision"].sum()),
            "fill_groups_with_missing_counterparty_rows": int((~dm["fill_group_consistent"]).sum()),
            "settlement": {k: settlement[mid].get(k) for k in ("status", "payout_per_share", "payout_token1", "payout_token2", "settlement_time_utc", "resolution_block", "resolution_tx", "oracle", "gamma_outcome_prices_string", "end_date_metadata")},
        }
        checks[mid]["sums_consistent"] = abs(checks[mid]["sum_decision_payoff_usd"] - checks[mid]["sum_actor_summary_usd"]) < 1e-3

    d_out = d[DECISION_COLS].copy()
    d_out["settlement_timestamp"] = pd.to_numeric(d_out["settlement_timestamp"], errors="coerce").astype("Int64")
    d_out["label_available_at"] = pd.to_numeric(d_out["label_available_at"], errors="coerce").astype("Int64")
    d_out.to_parquet(out / "market_decisions.parquet", index=False)
    summary.to_parquet(out / "actor_summary.parquet", index=False)
    all_mk.to_parquet(out / "actor_summary_all_markets.parquet", index=False)
    export = d_out.drop(columns=["datetime_utc"]).copy()
    with gzip.open(out / "market_decisions.jsonl.gz", "wt") as fh:
        for rec in export.to_dict(orient="records"):
            fh.write(json.dumps({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in rec.items()}, default=lambda o: None if pd.isna(o) else str(o)) + "\n")
    write_trajectories(d_out, out / "actor_trajectories.jsonl.gz")
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {"dataset": ext["dataset"], "revision": ext["revision"], "trades_parquet_sha256": ext.get("sha256_trades_parquet"),
                   "orderfilled_selected_sha256": fills_man.get("sha256_orderfilled_selected_parquet"),
                   "orderfilled_source_lfs_sha256": fills_man.get("source_lfs_sha256"), "exchange_addresses_treated_as_exchange": fills_man.get("exchange_addresses_detected")},
        "semantics_evidence": {
            "v1": "OrderFilled(maker, taker, makerAssetId, takerAssetId, makerAmountFilled, takerAmountFilled): the collateral leg is asset id 0; the event owner is `maker`; the order's own event has taker == exchange contract. Cross-check: for every (block, taker wallet) group the maker fills' share total equals the taker event's shares (see fill_groups_*).",
            "v2": "CTF_EXCHANGE_V2 events carry a side flag (0 BUY / 1 SELL) in the first id field and the token id in the second. Established from the Polygon receipt of tx 0xa6b35cae4c8d23096c4290cfb46711b6a7acc8b26e7e3d6211c0bd1d4d4e9580 (block 88019603): ERC-1155 transfers of 7.98 YES and 7.98 NO into the exchange, collateral of 4.08985+0.05975 fee and 3.8304 paid out, i.e. a MERGE of two SELL orders whose events have the flag 1.",
            "trades_parquet_caveat": "Wang's trades.parquet keeps only maker-order events, mirrors the maker side onto the taker (wrong for MINT/MERGE fills), rounds amounts to 2 dp, and for CTF_EXCHANGE_V2 drops every SELL order (side flag 1). All of that is bypassed by scoring from orderfilled events.",
            "order_hash_caveat": "order_hash is retained when present and can link partial executions of the same order. Rows are not collapsed: OrderFilled data omits unfilled and cancelled orders, and a maker order's first observed fill is only an upper bound on its placement time. Legacy extracts without order_hash remain supported with null values.",
        },
        "normalization": norm_rep, "validation_against_trades_parquet": val,
        "fee_treatment": "decision_payoff_usd is before fees. fee_amount is the collateral fee charged to the order owner (V2 events only; V1 fees are zero here). Verified on chain: V2 BUY at block 87994720 log 501 paid 10.06 for shares booked as 10.00 (fee 0.06 on top); V2 SELL in tx 0xa6b35cae... received 4.08985 of 4.1496 booked (fee 0.05975 deducted). decision_payoff_after_fee_usd = decision_payoff_usd - fee_amount.",
        "per_market": checks,
        "totals": {"decisions": int(len(d)), "scored": int(d["score_status"].eq("scored").sum()), "actors": int(summary["actor_wallet"].nunique())},
        "outputs": {p.name: sha256_file(p) for p in [out / "market_decisions.parquet", out / "actor_summary.parquet", out / "actor_summary_all_markets.parquet", out / "market_decisions.jsonl.gz", out / "actor_trajectories.jsonl.gz"]},
        "actor_id": "int64, first 48 bits of sha256(lowercased wallet); actor_seq orders each actor's decisions across the imported markets, actor_market_seq within one market",
        "columns_market_decisions": DECISION_COLS,
    }
    (out / "scoring_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps({"normalization": norm_rep, "validation": val}, indent=1))
    for mid, c in checks.items():
        print(f"{mid}: decisions {c['decisions']:,} scored {c['scored']:,} unscored {c['unscored_by_reason']} actors {c['actors']:,} | payout token1={c['settlement']['payout_token1']} at {c['settlement']['settlement_time_utc']} | sums consistent {c['sums_consistent']} | market-wide sum {c['market_wide_payoff_sum_usd']}")
    print(f"wrote {out}/market_decisions.parquet ({len(d):,} rows), actor_summary.parquet ({len(summary):,} rows), actor_summary_all_markets.parquet, market_decisions.jsonl.gz, actor_trajectories.jsonl.gz ({d['actor_id'].nunique():,} actors), scoring_manifest.json")


TRAJ_COLS = ["actor_seq", "actor_market_seq", "timestamp", "market_id", "actor_role", "action", "outcome_token", "outcome_label",
             "shares", "execution_price", "cash_amount_usd", "fee_amount", "decision_payoff_usd", "score_status",
             "label_available_at", "label_public_at_decision", "decision_id"]


def write_trajectories(d: pd.DataFrame, path: Path):
    """One line per actor: {"actor_id", "actor_wallet", "n_decisions", "markets", "decisions": [...]} in execution order."""
    d = d.sort_values(["actor_id", "actor_seq"])
    with gzip.open(path, "wt") as fh:
        for (aid, w), g in d.groupby(["actor_id", "actor_wallet"], sort=False):
            recs = g[TRAJ_COLS].to_dict(orient="records")
            for r in recs:
                for k, v in list(r.items()):
                    if isinstance(v, float) and np.isnan(v): r[k] = None
                    elif hasattr(v, "item"): r[k] = v.item()
            fh.write(json.dumps({"actor_id": int(aid), "actor_wallet": w, "n_decisions": len(recs),
                                 "markets": sorted(g["market_id"].unique().tolist()), "decisions": recs}) + "\n")


def cmd_trajectory(args):
    d = pd.read_parquet(Path(args.out) / "market_decisions.parquet")
    key = d["actor_wallet"].str.lower().eq(args.actor.lower()) if args.actor.startswith("0x") else d["actor_id"].eq(int(args.actor))
    g = d[key].sort_values("actor_seq")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 2000)
    print(g[["actor_seq", "datetime_utc", "market_id", "actor_role", "action", "outcome_label", "shares", "execution_price", "cash_amount_usd", "decision_payoff_usd"]].to_string(index=False))
    print(f"\nactor_id {g.actor_id.iloc[0]}  wallet {g.actor_wallet.iloc[0]}  decisions {len(g)}  markets {sorted(g.market_id.unique())}  sum payoff {g.decision_payoff_usd.sum():.2f}")


def cmd_wallet(args):
    out = Path(args.out)
    d = pd.read_parquet(out / "market_decisions.parquet"); s = pd.read_parquet(out / "actor_summary.parquet")
    w = args.wallet.lower()
    d = d[d["actor_wallet"].str.lower().eq(w)]; s = s[s["actor_wallet"].str.lower().eq(w)]
    if args.market: d = d[d["market_id"].eq(args.market)]; s = s[s["market_id"].eq(args.market)]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    cols = ["datetime_utc", "market_id", "actor_id", "actor_role", "action", "outcome_label", "shares", "execution_price", "cash_amount_usd", "settlement_payout", "decision_payoff_usd", "score_status"]
    print(d[cols].to_string(index=False))
    print("\nsummary:"); print(s.T.to_string(header=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("run"); a.add_argument("--selected", default="data/selected"); a.add_argument("--settlement", default="data/settlement"); a.add_argument("--out", default="data/scored"); a.set_defaults(fn=cmd_run)
    a = sp.add_parser("trajectory"); a.add_argument("--actor", required=True, help="wallet address or actor_id"); a.add_argument("--out", default="data/scored"); a.set_defaults(fn=cmd_trajectory)
    a = sp.add_parser("wallet"); a.add_argument("--wallet", required=True); a.add_argument("--market", default=None); a.add_argument("--out", default="data/scored"); a.set_defaults(fn=cmd_wallet)
    args = p.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
