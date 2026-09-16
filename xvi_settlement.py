"""
xvi_settlement.py - authoritative settlement payouts for Polymarket conditions.

Source of truth: the Gnosis ConditionalTokens contract on Polygon
(0x4D97DCd97eC945f40cF65F87097ACe5EA0476045). For each condition_id we read

  payoutDenominator(conditionId)            0 until resolved
  payoutNumerators(conditionId, i)          i = 0 .. outcomeSlotCount-1

and locate the ConditionResolution event (block, timestamp, oracle, tx hash) by
bisecting payoutDenominator over block height on an archive node and then
fetching logs in a narrow window. We also verify which outcome index each
traded token corresponds to by recomputing the ERC-1155 position id with
getCollectionId / getPositionId (collateral USDC.e for the plain exchange,
WrappedCollateral for negative-risk markets).

Nothing here uses market titles, end_date, last trade prices or Gamma's
outcome_prices string as the payout. Results are cached as JSON per condition.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from pathlib import Path

import requests
from Crypto.Hash import keccak

CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
COLLATERALS = {
    "USDC.e": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    "WrappedCollateral(neg-risk)": "0x3A3BD7bb9528E159577F7C2e685CC81A765002E2",
}
ARCHIVE_RPCS = [os.environ.get("POLYGON_RPC"), "https://polygon.drpc.org"]
CTF_LAUNCH_BLOCK = 30_000_000  # well before the first Polymarket CLOB trade (Nov 2022, block ~35.9M)


def kec(s: str) -> str:
    k = keccak.new(digest_bits=256); k.update(s.encode()); return k.hexdigest()


SEL = {n: kec(n + sig)[:8] for n, sig in [
    ("payoutDenominator", "(bytes32)"), ("payoutNumerators", "(bytes32,uint256)"),
    ("getCollectionId", "(bytes32,bytes32,uint256)"), ("getPositionId", "(address,bytes32)"),
    ("getOutcomeSlotCount", "(bytes32)")]}
TOPIC_RESOLUTION = "0x" + kec("ConditionResolution(bytes32,address,bytes32,uint256,uint256[])")


def w(x) -> str:
    """32-byte ABI word."""
    if isinstance(x, int): return hex(x)[2:].rjust(64, "0")
    return x[2:].lower().rjust(64, "0") if x.startswith("0x") else x.lower().rjust(64, "0")


class Rpc:
    def __init__(self, urls=None):
        self.urls = [u for u in (urls or ARCHIVE_RPCS) if u]
        self.s = requests.Session(); self.url = self.urls[0]; self.calls = 0

    def call(self, method, params):
        last = None
        for url in [self.url] + [u for u in self.urls if u != self.url]:
            for attempt in range(4):
                try:
                    r = self.s.post(url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=60).json()
                    self.calls += 1
                    if "result" in r and r["result"] is not None:
                        self.url = url; return r["result"]
                    last = r.get("error", r)
                    if isinstance(last, dict) and "rate" in str(last).lower(): time.sleep(2 ** attempt); continue
                    break
                except requests.RequestException as e:
                    last = repr(e); time.sleep(2 ** attempt)
        raise RuntimeError(f"rpc {method} failed: {last}")

    def eth_call(self, to, data, block="latest") -> str:
        return self.call("eth_call", [{"to": to, "data": data}, block if isinstance(block, str) else hex(block)])

    def u256(self, to, data, block="latest") -> int:
        return int(self.eth_call(to, data, block), 16)


def verify_token_mapping(rpc: Rpc, condition_id: str, token1: str, token2: str) -> dict:
    """Which outcome index (0/1) does each traded token id encode? Recomputed on chain."""
    out = {"token1_outcome_index": None, "token2_outcome_index": None, "collateral": None, "verified": False}
    for cname, caddr in COLLATERALS.items():
        found = {}
        for index_set in (1, 2):  # bit i set => outcome index i-1
            coll = rpc.eth_call(CTF, "0x" + SEL["getCollectionId"] + w(0) + w(condition_id) + w(index_set))
            pos = rpc.u256(CTF, "0x" + SEL["getPositionId"] + w(caddr) + coll[2:])
            found[str(pos)] = index_set - 1
        if token1 in found and token2 in found:
            out.update(token1_outcome_index=found[token1], token2_outcome_index=found[token2], collateral=cname, verified=True)
            return out
    return out


def find_resolution_block(rpc: Rpc, condition_id: str, latest: int) -> int | None:
    data = "0x" + SEL["payoutDenominator"] + w(condition_id)
    if rpc.u256(CTF, data, latest) == 0:
        return None
    lo, hi = CTF_LAUNCH_BLOCK, latest
    while lo < hi:
        mid = (lo + hi) // 2
        if rpc.u256(CTF, data, mid) == 0: lo = mid + 1
        else: hi = mid
    return lo


def fetch_settlement(rpc: Rpc, market: dict) -> dict:
    cond = market["condition_id"]
    rec = {"market_id": market["id"], "condition_id": cond, "question": market["question"],
           "settlement_source": f"Polygon ConditionalTokens {CTF} via {rpc.url}",
           "payout_verified_at": dt.datetime.now(dt.timezone.utc).isoformat(), "settlement_verified": False}
    rec["token_mapping"] = verify_token_mapping(rpc, cond, market["token1"], market["token2"])
    slots = rpc.u256(CTF, "0x" + SEL["getOutcomeSlotCount"] + w(cond))
    den = rpc.u256(CTF, "0x" + SEL["payoutDenominator"] + w(cond))
    rec.update(outcome_slot_count=slots, payout_denominator=den)
    if den == 0:
        rec["status"] = "unresolved_on_chain"; return rec
    nums = [rpc.u256(CTF, "0x" + SEL["payoutNumerators"] + w(cond) + w(i)) for i in range(slots)]
    rec["payout_numerators"] = nums
    rec["payout_per_share"] = [n / den for n in nums]
    latest = int(rpc.call("eth_blockNumber", []), 16)
    blk = find_resolution_block(rpc, cond, latest)
    rec["resolution_block_by_bisection"] = blk
    logs = rpc.call("eth_getLogs", [{"address": CTF, "topics": [TOPIC_RESOLUTION, cond], "fromBlock": hex(max(blk - 50, 0)), "toBlock": hex(blk + 50)}])
    if isinstance(logs, list) and logs:
        l = logs[-1]; data = l["data"][2:]; words = [int(data[i:i + 64], 16) for i in range(0, len(data), 64)]
        ev_nums = words[3:3 + words[2]]
        b = rpc.call("eth_getBlockByNumber", [l["blockNumber"], False])
        ts = int(b["timestamp"], 16)
        rec.update(resolution_block=int(l["blockNumber"], 16), resolution_tx=l["transactionHash"], oracle="0x" + l["topics"][2][-40:],
                   event_payout_numerators=ev_nums, settlement_timestamp=ts,
                   settlement_time_utc=dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(),
                   event_matches_state=(ev_nums == nums))
        rec["settlement_verified"] = bool(rec["token_mapping"]["verified"] and ev_nums == nums)
        rec["status"] = "resolved" if rec["settlement_verified"] else "resolved_but_inconsistent"
    else:
        rec["status"] = "resolved_state_but_event_not_found"
        rec["settlement_verified"] = bool(rec["token_mapping"]["verified"])  # payout known; timestamp not established
    # cross-check only, never used as the payout
    rec["gamma_outcome_prices_string"] = market.get("outcome_prices")
    rec["end_date_metadata"] = str(market.get("end_date"))
    if rec["token_mapping"]["verified"]:
        rec["payout_token1"] = rec["payout_per_share"][rec["token_mapping"]["token1_outcome_index"]]
        rec["payout_token2"] = rec["payout_per_share"][rec["token_mapping"]["token2_outcome_index"]]
    rec["rpc_calls"] = rpc.calls
    return rec


def load_or_fetch(markets: list[dict], cache_dir: Path, refresh: bool = False, rpc: Rpc | None = None) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for m in markets:
        p = cache_dir / f"{m['condition_id']}.json"
        if p.exists() and not refresh:
            out[m["id"]] = json.loads(p.read_text()); continue
        rpc = rpc or Rpc()
        rec = fetch_settlement(rpc, m)
        p.write_text(json.dumps(rec, indent=2))
        out[m["id"]] = rec
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selected", default="data/selected"); ap.add_argument("--cache", default="data/settlement"); ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    mk = json.loads((Path(a.selected) / "markets.json").read_text())
    for mid, rec in load_or_fetch(mk, Path(a.cache), a.refresh).items():
        print(mid, rec["status"], "payouts", rec.get("payout_per_share"), "token1->", rec.get("payout_token1"), "at", rec.get("settlement_time_utc"), "verified", rec["settlement_verified"])
