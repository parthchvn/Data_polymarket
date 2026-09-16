from __future__ import annotations

import json

import pandas as pd
import pytest

from filter_human_trades import (
    LABEL_HUMAN,
    LABEL_INSUFFICIENT,
    LABEL_LIKELY,
    LABEL_REVIEW,
    LITERATURE_SYSTEMATIC_ARBITRAGE_WALLETS,
    _load_overrides,
    _validate_and_prepare,
    build_actor_market_features,
    classify_actor_markets,
    classify_decisions,
    compute_thresholds,
    run_to_directory,
)


def decision(
    wallet: str,
    i: int,
    *,
    market: str = "m1",
    role: str = "taker",
    day: int = 0,
    second: int | None = None,
    block: int | None = None,
    action: str = "BUY",
    token: str = "token1",
    shares: float = 1.0,
    counterparty: str | None = None,
    order_hash: str | None = None,
) -> dict:
    second = i * 60 if second is None else second
    block = i + 1 if block is None else block
    row = {
        "decision_id": f"{market}-{wallet.lower()}-{i}",
        "actor_wallet": wallet,
        "market_id": market,
        "actor_role": role,
        "timestamp": 1_700_000_000 + day * 86_400 + second,
        "block_number": block,
        "log_index": i,
        "action": action,
        "outcome_token": token,
        "shares": shares,
        "counterparty_wallet": counterparty if role == "maker" else None,
    }
    if order_hash is not None:
        row["order_hash"] = order_hash
    return row


def make_human(wallet: str = "0xhuman", *, market: str = "m1", start: int = 0) -> list[dict]:
    return [
        decision(wallet, start + i, market=market, day=i, block=start + i + 1)
        for i in range(20)
    ]


def make_market_maker(wallet: str = "0xmm", *, market: str = "m1") -> list[dict]:
    rows = []
    for i in range(180):
        plus = i % 2 == 0
        rows.append(
            decision(
                wallet,
                i,
                market=market,
                role="maker",
                day=i % 6,
                second=(i // 6) * 60,
                block=10_000 + i,
                action="BUY" if plus else "SELL",
                token="token1",
                counterparty=f"0xcp{i % 60:02x}",
                order_hash=f"0xorder{i:04x}",
            )
        )
    return rows


def make_fast_taker(wallet: str = "0xhft", *, market: str = "m1") -> list[dict]:
    rows = []
    for i in range(240):
        day = i // 80
        within_day = i % 80
        rows.append(
            decision(
                wallet,
                i,
                market=market,
                role="taker",
                day=day,
                second=within_day * 2,
                block=20_000 + i,
            )
        )
    return rows


def test_casefold_low_history_and_no_label_leakage():
    rows = [
        decision("0xAbC", 0, block=1),
        decision("0xaBc", 1, day=1, block=2),
    ]
    base = pd.DataFrame(rows)
    base["decision_payoff_usd"] = [1000.0, -999.0]
    base["settlement_payout"] = [1.0, 0.0]
    first = classify_decisions(base, include_literature_exclusions=False)
    changed = base.copy()
    changed["decision_payoff_usd"] *= -123
    changed["settlement_payout"] = 1 - changed["settlement_payout"]
    second = classify_decisions(changed, include_literature_exclusions=False)
    assert len(first["wallet_features"]) == 1
    assert first["wallet_features"].iloc[0].wallet_classification == LABEL_INSUFFICIENT
    pd.testing.assert_series_equal(
        first["classified"]["row_classification"],
        second["classified"]["row_classification"],
    )


def test_exact_duplicate_drops_but_conflicting_id_fails():
    row = decision("0xa", 0)
    clean, dropped = _validate_and_prepare(pd.DataFrame([row, row]), source_name="test")
    assert len(clean) == 1 and dropped == 1
    conflict = dict(row)
    conflict["shares"] = 2.0
    with pytest.raises(ValueError, match="conflicting rows"):
        _validate_and_prepare(pd.DataFrame([row, conflict]), source_name="test")


def test_order_hash_activity_units_and_block_fallback():
    rows = [
        decision("0xa", 0, role="maker", block=10, counterparty="0xc1", order_hash="0xH1"),
        decision("0xa", 1, role="maker", block=11, counterparty="0xc2", order_hash="0xH1"),
        decision("0xa", 2, role="maker", block=11, counterparty="0xc3", order_hash="0xH2"),
        decision("0xa", 3, role="maker", block=12, counterparty="0xc4"),
        decision("0xa", 4, role="maker", block=12, counterparty="0xc5"),
    ]
    # Ensure the optional column exists for rows where it was omitted.
    frame = pd.DataFrame(rows)
    features = build_actor_market_features(frame).iloc[0]
    assert features.maker_blocks == 3
    assert features.maker_activity_units == 3  # H1, H2, and one block fallback unit
    assert features.maker_order_hash_coverage == pytest.approx(3 / 5)
    assert features.maker_activity_unit_mode == "mixed_order_hash_block_fallback"


def test_fragmented_maker_fills_do_not_create_hft_signal():
    rows = [
        decision(
            "0xmaker",
            i,
            role="maker",
            block=42,
            second=0,
            counterparty=f"0xc{i}",
            action="BUY" if i % 2 == 0 else "SELL",
        )
        for i in range(300)
    ]
    frame = pd.DataFrame(rows)
    features = build_actor_market_features(frame)
    classified = classify_actor_markets(features, compute_thresholds(features)).iloc[0]
    assert classified.maker_blocks == 1
    assert classified.maker_activity_units == 1
    assert classified.taker_blocks == 0
    assert not classified.strong_high_frequency_taker_footprint
    assert not classified.strong_market_maker_footprint


def test_strong_market_maker_and_directional_whale():
    mm = make_market_maker()
    one_sided = []
    for row in make_market_maker("0xwhale"):
        row["action"] = "BUY"
        one_sided.append(row)
    frame, _ = _validate_and_prepare(pd.DataFrame(mm + one_sided), source_name="test")
    features = build_actor_market_features(frame)
    classified = classify_actor_markets(features, compute_thresholds(features)).set_index(
        "actor_wallet"
    )
    assert classified.loc["0xmm", "strong_market_maker_footprint"]
    assert not classified.loc["0xwhale", "strong_market_maker_footprint"]
    assert classified.loc["0xwhale", "maker_flow_balance"] == 0


def test_taker_only_timing_flags_fast_taker_not_maker_or_same_block_sweep():
    fast = make_fast_taker()
    passive = []
    for row in make_fast_taker("0xpassive"):
        row["actor_role"] = "maker"
        row["counterparty_wallet"] = "0xcp"
        passive.append(row)
    sweep = [
        decision("0xsweep", i, block=999, second=0, role="taker") for i in range(300)
    ]
    frame, _ = _validate_and_prepare(pd.DataFrame(fast + passive + sweep), source_name="test")
    features = build_actor_market_features(frame)
    classified = classify_actor_markets(features, compute_thresholds(features)).set_index(
        "actor_wallet"
    )
    assert classified.loc["0xhft", "strong_high_frequency_taker_footprint"]
    assert not classified.loc["0xpassive", "strong_high_frequency_taker_footprint"]
    assert classified.loc["0xsweep", "taker_blocks"] == 1
    assert not classified.loc["0xsweep", "strong_high_frequency_taker_footprint"]


def test_single_taker_timing_signal_is_review_not_human_or_excluded():
    rows = []
    for i in range(240):
        day = i // 80
        within_day = i % 80
        # Eight blocks in each of ten hours: high daily rate, but no five-second
        # burst, hourly burst, 100-block day, or 12-hour coverage signal.
        second = (within_day // 8) * 3_600 + (within_day % 8) * 60
        rows.append(decision("0xreview", i, day=day, second=second, block=30_000 + i))
    result = classify_decisions(
        pd.DataFrame(rows), profile="lenient", include_literature_exclusions=False
    )
    feature = result["actor_market_features"].iloc[0]
    assert feature.taker_timing_signal_count == 1
    assert feature.review_high_frequency_taker_footprint
    assert result["wallet_features"].iloc[0].wallet_classification == LABEL_REVIEW
    assert len(result["partitions"]["uncertain"]) == 240


def test_published_activity_screen_uses_exact_greater_than():
    exact = [decision("0xexact", i, block=i + 1) for i in range(20)]
    over = [decision("0xover", i, block=100 + i) for i in range(21)]
    result = classify_decisions(
        pd.DataFrame(exact + over), profile="strict", include_literature_exclusions=False
    )
    wallets = result["wallet_features"].set_index("actor_wallet")
    assert not wallets.loc["0xexact", "published_activity_screen"]
    assert wallets.loc["0xover", "published_activity_screen"]
    assert wallets.loc["0xover", "wallet_classification"] == LABEL_LIKELY


def test_partitions_are_disjoint_exhaustive_and_human_output_is_pure():
    literature = next(iter(LITERATURE_SYSTEMATIC_ARBITRAGE_WALLETS))
    rows = make_human() + [decision("0xlow", 100)] + [decision(literature, 200)]
    result = classify_decisions(pd.DataFrame(rows), profile="lenient")
    parts = result["partitions"]
    ids = [set(x.decision_id) for x in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == set(result["classified"].decision_id)
    assert set(parts["human_candidate"].row_classification) == {LABEL_HUMAN}
    assert set(parts["uncertain"].row_classification) <= {LABEL_REVIEW, LABEL_INSUFFICIENT}
    assert set(parts["excluded"].row_classification) == {LABEL_LIKELY}


def test_row_order_does_not_change_wallet_labels():
    frame = pd.DataFrame(make_human("0xhuman") + make_fast_taker("0xhft"))
    first = classify_decisions(frame, profile="lenient", include_literature_exclusions=False)
    shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    second = classify_decisions(
        shuffled, profile="lenient", include_literature_exclusions=False
    )
    a = first["wallet_features"].set_index("actor_wallet")["wallet_classification"].sort_index()
    b = second["wallet_features"].set_index("actor_wallet")["wallet_classification"].sort_index()
    pd.testing.assert_series_equal(a, b)


def test_market_scope_localizes_behavioral_exclusion():
    rows = make_market_maker("0xmulti", market="a") + make_human(
        "0xmulti", market="b", start=1_000
    )
    frame = pd.DataFrame(rows)
    wallet_scope = classify_decisions(
        frame, profile="lenient", scope="wallet", include_literature_exclusions=False
    )
    market_scope = classify_decisions(
        frame, profile="lenient", scope="market", include_literature_exclusions=False
    )
    assert set(wallet_scope["classified"].row_classification) == {LABEL_LIKELY}
    per_market = market_scope["classified"].groupby("market_id").row_classification.unique()
    assert set(per_market["a"]) == {LABEL_LIKELY}
    assert set(per_market["b"]) == {LABEL_HUMAN}


def test_external_reference_controls_activity_screen():
    input_rows = [decision("0xactive", 0)]
    reference_rows = []
    for i in range(501):
        reference_rows.append(
            decision("0xactive", i, day=i // 10, second=(i % 10) * 60, block=10_000 + i)
        )
    result = classify_decisions(
        pd.DataFrame(input_rows),
        reference_decisions=pd.DataFrame(reference_rows),
        profile="strict",
        include_literature_exclusions=False,
    )
    wallet = result["wallet_features"].iloc[0]
    assert wallet.activity_total_decisions == 501
    assert wallet.published_activity_screen
    assert result["reference_scope"] == "external_reference"


def test_reviewed_override_has_final_say(tmp_path):
    literature = next(iter(LITERATURE_SYSTEMATIC_ARBITRAGE_WALLETS))
    frame = pd.DataFrame([decision(literature, 0)])
    override_path = tmp_path / "overrides.csv"
    pd.DataFrame(
        [{"wallet": literature.upper(), "label": "include", "reason": "verified owner"}]
    ).to_csv(override_path, index=False)
    result = classify_decisions(frame, overrides=_load_overrides(override_path))
    assert result["wallet_features"].iloc[0].wallet_classification == LABEL_HUMAN
    assert result["classified"].iloc[0].row_classification == LABEL_HUMAN


def test_cli_outputs_manifest_and_hashes(tmp_path):
    input_path = tmp_path / "input.parquet"
    out = tmp_path / "out"
    pd.DataFrame(make_human() + [decision("0xlow", 999)]).to_parquet(input_path, index=False)
    result = run_to_directory(
        input_path,
        out,
        profile="lenient",
        include_literature_exclusions=False,
    )
    expected = {
        "classified_decisions.parquet",
        "human_candidate_decisions.parquet",
        "uncertain_decisions.parquet",
        "excluded_decisions.parquet",
        "actor_market_features.parquet",
        "wallet_features.parquet",
        "wallet_classification.csv",
        "thresholds.json",
        "classification_manifest.json",
    }
    assert expected <= {p.name for p in out.iterdir()}
    manifest = json.loads((out / "classification_manifest.json").read_text())
    assert manifest["guardrails"]["settlement_payoff_or_correctness_used"] is False
    assert manifest["reference"]["left_censored_to_imported_markets"] is True
    assert sum(
        manifest["counts"][key]
        for key in ("human_candidate_rows", "uncertain_rows", "excluded_rows")
    ) == manifest["counts"]["classified_rows"]
    assert result["output_paths"]["manifest"].exists()
