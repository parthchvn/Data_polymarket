"""Synthetic-fixture tests for xvi_score. None of these rows are historical data."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xvi_score import aggregate_actors, normalize_fills, score_decisions, add_actor_sequences, actor_id_of, DECISION_COLS  # noqa: E402

T1, T2 = "111", "222"
MARKETS = [{"id": "M", "condition_id": "0xc", "token1": T1, "token2": T2, "answer1": "Yes", "answer2": "No"}]
EX = "0xEXCHANGE"


def settle(v1, v2=None, verified=True, ts=2_000_000):
    v2 = (1 - v1 if v1 is not None else None) if v2 is None else v2
    return {"M": {"settlement_verified": verified, "payout_token1": v1, "payout_token2": v2, "settlement_timestamp": ts,
                  "settlement_source": "test", "payout_verified_at": "test", "status": "resolved" if verified else "unresolved_on_chain"}}


_n = [0]


def fill(wallet, action, token, shares, price, *, role="taker", contract="NEGRISK_CTF_EXCHANGE", counterparty="0xCP", block=100, log=None, ts=1_000_000, fee=0, v2=False):
    """Build one OrderFilled row in the orderfilled_selected.parquet layout."""
    _n[0] += 1
    log = _n[0] if log is None else log
    q, u = int(round(shares * 1e6)), int(round(shares * price * 1e6))
    if v2:
        contract = "CTF_EXCHANGE_V2"
        ma, ta = ("0", token) if action == "BUY" else ("1", token)
        mam, tam = (u, q) if action == "BUY" else (q, u)
    else:
        ma, ta = ("0", token) if action == "BUY" else (token, "0")
        mam, tam = (u, q) if action == "BUY" else (q, u)
    return {"timestamp": ts, "block_number": block, "log_index": log, "contract": contract, "maker": wallet,
            "taker": EX if role == "taker" else counterparty, "maker_asset_id": ma, "taker_asset_id": ta,
            "maker_amount_filled": mam, "taker_amount_filled": tam, "maker_fee": int(fee * 1e6), "taker_fee": 0, "protocol_fee": 0,
            "event_role": "taker_order" if role == "taker" else "maker_order", "asset_id": token, "market_id": "M",
            "nonusdc_side": "token1" if token == T1 else "token2", "in_trades_parquet": role == "maker"}


def run(rows, settlement):
    d, rep = normalize_fills(pd.DataFrame(rows), MARKETS, None, "rev")
    return score_decisions(d, settlement), rep


def payoff(d, wallet):
    return d.loc[d.actor_wallet.eq(wallet), "decision_payoff_usd"].tolist()


def test_buy_and_sell_on_winning_and_losing_tokens():
    rows = [fill("A", "BUY", T1, 100, 0.4), fill("B", "SELL", T1, 100, 0.4), fill("C", "BUY", T2, 100, 0.6), fill("D", "SELL", T2, 100, 0.6)]
    d, _ = run(rows, settle(1.0))  # YES wins
    assert payoff(d, "A") == [60.0] and payoff(d, "B") == [-60.0] and payoff(d, "C") == [-60.0] and payoff(d, "D") == [60.0]
    d, _ = run(rows, settle(0.0))  # NO wins
    assert payoff(d, "A") == [-40.0] and payoff(d, "B") == [40.0] and payoff(d, "C") == [40.0] and payoff(d, "D") == [-40.0]
    assert (d.score_status == "scored").all()


def test_fractional_settlement_payout():
    d, _ = run([fill("A", "BUY", T1, 100, 0.4), fill("B", "SELL", T2, 100, 0.7)], settle(0.5, 0.5))
    assert payoff(d, "A") == [pytest.approx(10.0)]   # 100*0.5 - 40
    assert payoff(d, "B") == [pytest.approx(20.0)]   # 70 - 100*0.5
    assert (d.settlement_payout == 0.5).all()


def test_spec_example_buy_040_sell_070_is_plus_30_under_both_outcomes():
    rows = [fill("A", "BUY", T1, 100, 0.4, ts=1), fill("A", "SELL", T1, 100, 0.7, ts=2, block=101)]
    for v, expect in ((1.0, [60.0, -30.0]), (0.0, [-40.0, 70.0])):
        d, _ = run(rows, settle(v))
        assert payoff(d, "A") == expect
        s = aggregate_actors(d)
        assert s.actor_market_settlement_payoff_usd.iloc[0] == pytest.approx(30.0)
        assert s.net_position_token1.iloc[0] == 0 and not s.external_inventory_detected.iloc[0]


def test_partial_sale_and_shares_held_through_settlement():
    rows = [fill("A", "BUY", T1, 100, 0.4, ts=1), fill("A", "SELL", T1, 40, 0.7, ts=2, block=101)]
    d, _ = run(rows, settle(1.0))
    assert payoff(d, "A") == [60.0, -12.0]
    s = aggregate_actors(d)
    # 40 sold for a locked 0.30 gain, 60 held to a $1 payout: 40*0.3 + 60*0.6 = 48; redemption is not added again
    assert s.actor_market_settlement_payoff_usd.iloc[0] == pytest.approx(48.0)
    assert s.net_position_token1.iloc[0] == pytest.approx(60.0)


def test_original_token_role_and_mint_semantics_are_preserved():
    # taker buys YES; matched NORMAL against maker selling YES and MINT against maker buying NO. Nobody is mirrored.
    rows = [fill("MK1", "SELL", T1, 10, 0.37, role="maker", counterparty="TK", log=1),
            fill("MK2", "BUY", T2, 20, 0.63, role="maker", counterparty="TK", log=2),
            fill("TK", "BUY", T1, 30, 0.37, role="taker", log=3)]
    d, rep = run(rows, settle(1.0))
    r = d.set_index("actor_wallet")
    assert r.loc["MK2", "action"] == "BUY" and r.loc["MK2", "outcome_token"] == "token2" and r.loc["MK2", "outcome_label"] == "No"
    assert r.loc["MK2", "execution_price"] == pytest.approx(0.63)   # original NO price, not 1-0.37
    assert r.loc["TK", "action"] == "BUY" and r.loc["TK", "outcome_token"] == "token1" and r.loc["TK", "actor_role"] == "taker"
    assert r.loc["MK1", "counterparty_wallet"] == "TK" and pd.isna(r.loc["TK", "counterparty_wallet"])
    assert rep["fill_groups_consistent"] == 1 and d.fill_group_consistent.all()
    # zero-sum before fees within the market
    assert d.decision_payoff_usd.sum() == pytest.approx(0.0)


def test_v2_side_flag_rows_decode_as_buy_and_sell():
    rows = [fill("A", "BUY", T1, 12.5, 0.8, v2=True, fee=0.06), fill("B", "SELL", T2, 7.98, 0.48, v2=True, fee=0.05)]
    d, _ = run(rows, settle(1.0))
    r = d.set_index("actor_wallet")
    assert r.loc["A", "action"] == "BUY" and r.loc["A", "shares"] == pytest.approx(12.5) and r.loc["A", "cash_amount_usd"] == pytest.approx(10.0)
    assert r.loc["B", "action"] == "SELL" and r.loc["B", "shares"] == pytest.approx(7.98) and r.loc["B", "execution_price"] == pytest.approx(0.48)
    assert r.loc["A", "decision_payoff_usd"] == pytest.approx(2.5) and r.loc["A", "decision_payoff_after_fee_usd"] == pytest.approx(2.44)
    assert (r.fee_asset == "usd").all()


def test_duplicates_removed_but_multiple_fills_in_one_transaction_kept():
    a = fill("A", "BUY", T1, 10, 0.5, role="maker", block=7, log=5)
    rows = [a, dict(a), fill("A", "BUY", T1, 10, 0.5, role="maker", block=7, log=6), fill("TK", "BUY", T1, 20, 0.5, block=7, log=7)]
    d, rep = run(rows, settle(1.0))
    assert rep["duplicate_rows_removed"] == 1 and rep["decisions"] == 3
    assert (d[d.actor_wallet.eq("A")].decision_id.tolist() == ["7-5", "7-6"])
    assert aggregate_actors(d).set_index("actor_wallet").loc["A", "n_decisions"] == 2


def test_unknown_payout_and_unknown_semantics_are_not_scored_as_zero():
    d, _ = run([fill("A", "BUY", T1, 10, 0.5)], settle(None, None, verified=False))
    assert d.score_status.iloc[0] == "unavailable_settlement_unverified" and np.isnan(d.decision_payoff_usd.iloc[0])
    bad = fill("B", "BUY", T1, 10, 0.5); bad["contract"] = "UNKNOWN_EXCHANGE"   # event layout not established for this contract
    d, rep = run([fill("A", "BUY", T1, 10, 0.5), bad], settle(1.0))
    assert rep["action_unverified"] == 1
    r = d.set_index("actor_wallet")
    assert r.loc["B", "score_status"] == "unavailable_action_unverified" and np.isnan(r.loc["B", "decision_payoff_usd"])
    assert r.loc["A", "score_status"] == "scored"


def test_partial_coverage_is_reported_not_silently_summed():
    ok = fill("A", "BUY", T1, 10, 0.5); bad = fill("A", "BUY", T1, 10, 0.5, block=101); bad["contract"] = "UNKNOWN_EXCHANGE"
    d, _ = run([ok, bad], settle(1.0))
    s = aggregate_actors(d).iloc[0]
    assert s.n_decisions == 2 and s.n_scored == 1 and s.n_unscored == 1
    assert s.coverage_status.startswith("partial") and s.unscored_reasons == "unavailable_action_unverified"
    assert s.actor_market_settlement_payoff_usd == pytest.approx(5.0)


def test_external_inventory_is_flagged_when_selling_more_than_bought():
    d, _ = run([fill("A", "SELL", T1, 50, 0.9, ts=1)], settle(1.0))
    s = aggregate_actors(d).iloc[0]
    assert s.external_inventory_detected and s.min_running_position_token1 == pytest.approx(-50)


def test_label_leakage_flag_and_timestamps():
    d, _ = run([fill("A", "BUY", T1, 1, 0.5, ts=1_000), fill("A", "BUY", T1, 1, 0.5, ts=3_000, block=101)], settle(1.0, ts=2_000))
    assert d.label_public_at_decision.tolist() == [False, True]
    assert (d.label_available_at == 2_000).all() and d.timestamp.tolist() == [1_000, 3_000]


def test_decisions_summary_and_export_agree(tmp_path):
    rows = [fill("A", "BUY", T1, 100, 0.4, ts=1), fill("A", "SELL", T1, 40, 0.7, ts=2, block=101), fill("B", "SELL", T2, 5, 0.3, ts=3, block=102)]
    d, _ = run(rows, settle(1.0))
    d = add_actor_sequences(d)
    s = aggregate_actors(d)
    assert d.decision_payoff_usd.sum() == pytest.approx(s.actor_market_settlement_payoff_usd.sum())
    out = d[DECISION_COLS]
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for rec in out.drop(columns=["datetime_utc"]).to_dict(orient="records"):
            fh.write(json.dumps(rec, default=str) + "\n")
    back = pd.read_json(p, lines=True)
    assert len(back) == len(out) and back.decision_payoff_usd.sum() == pytest.approx(out.decision_payoff_usd.sum())


def test_actor_id_and_sequence_numbers():
    rows = [fill("A", "BUY", T1, 1, 0.5, ts=30, block=3), fill("B", "BUY", T1, 1, 0.5, ts=10, block=1), fill("A", "SELL", T1, 1, 0.6, ts=20, block=2)]
    d, _ = run(rows, settle(1.0))
    d = add_actor_sequences(d)
    a = d[d.actor_wallet.eq("A")].sort_values("actor_seq")
    assert a.actor_seq.tolist() == [1, 2] and a.action.tolist() == ["SELL", "BUY"] and a.timestamp.tolist() == [20, 30]
    assert a.prev_decision_timestamp.tolist()[1] == 20 and pd.isna(a.prev_decision_timestamp.iloc[0])
    assert d.actor_id.nunique() == 2 and actor_id_of("0xAbC") == actor_id_of("0xabc")
