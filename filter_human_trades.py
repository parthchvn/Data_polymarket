#!/usr/bin/env python3
"""Conservative, auditable filtering of automated-looking Polymarket wallets.

This module intentionally does not claim to prove that a retained wallet is human.
It separates decisions into three disjoint outputs:

* ``human_candidate_decisions.parquet``: enough observed history and no screen hit;
* ``uncertain_decisions.parquet``: review or insufficient-history wallets; and
* ``excluded_decisions.parquet``: strong behavioral, published-activity, literature,
  or reviewed-override exclusions.

Only execution-side fields are used. Settlement, payoff, correctness, and resolved
outcomes are not classifier inputs. Maker executions are not treated as evidence of
reaction speed because their timestamps record fills of resting orders, not the time
at which the orders were submitted or updated.

Example
-------
python filter_human_trades.py \
  --input data/scored/market_decisions.parquet \
  --out data/human_filter \
  --profile strict
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CLASSIFIER_VERSION = "1.0.0"
WILSON_Z_ONE_SIDED_95 = 1.6448536269514722

# Saguillo et al., arXiv:2508.03474. These are systematic-arbitrage labels from
# external research, not labels inferred by the rules in this file.
LITERATURE_SYSTEMATIC_ARBITRAGE_WALLETS = {
    "0xd218e474776403a330142299f7796e8ba32eb5c9",
    "0x63d43bbb87f85af03b8f2f9e2fad7b54334fa2f1",
    "0x9d84ce0306f8551e02efef1680475fc0f1dc1344",
    "0x44c1dfe43260c94ed4f1d00de2e1f80fb113ebc1",
    "0xb7d54bf1d0a362beb916d9cb58a04c41d67e0789",
    "0x53d2d3c78597a78402d4db455a680da7ef560c3f",
}

LITERATURE_SOURCE = {
    "label": "literature_identified_systematic_arbitrageur",
    "citation": "Saguillo et al., arXiv:2508.03474",
    "identifier": "arXiv:2508.03474",
}

# Della Vedova's published operational screen is balanced (50/1,000). Strict
# and lenient are declared sensitivity profiles around that screen. Comparisons
# are deliberately strict inequalities, matching "more than" in the source.
ACTIVITY_PROFILES = {
    "strict": {"peak_utc_day_gt": 20, "total_decisions_gt": 500},
    "balanced": {"peak_utc_day_gt": 50, "total_decisions_gt": 1_000},
    "lenient": {"peak_utc_day_gt": 100, "total_decisions_gt": 2_000},
}

DELLA_VEDOVA_SOURCE = {
    "name": "Della Vedova Prediction Market Indices: Algorithmic Share of Participation",
    "url": "https://jdellavedova.com/bot-share/",
    "published_rule": "more than 50 trades on the peak day or more than 1,000 lifetime trades",
}

LABEL_LIKELY = "likely_automated_or_mm"
LABEL_REVIEW = "review"
LABEL_INSUFFICIENT = "insufficient_history"
LABEL_HUMAN = "human_candidate"
VALID_LABELS = {LABEL_LIKELY, LABEL_REVIEW, LABEL_INSUFFICIENT, LABEL_HUMAN}

# This allow-list is the complete classifier feature boundary. Columns not in it
# may be carried into the classified outputs, but cannot affect a label.
REQUIRED_FEATURE_COLUMNS = [
    "decision_id",
    "actor_wallet",
    "market_id",
    "actor_role",
    "timestamp",
    "block_number",
    "log_index",
    "action",
    "outcome_token",
    "shares",
    "counterparty_wallet",
]

OPTIONAL_FEATURE_COLUMNS = ["order_hash"]

PROFILE_COLUMNS = ["decision_id", "actor_wallet", "timestamp"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _json_dump(data: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_scalar) + "\n",
        encoding="utf-8",
    )


def wilson_lower_bound(successes: pd.Series, trials: pd.Series) -> pd.Series:
    """One-sided 95% Wilson lower bound, with zero evidence mapped to zero."""
    k = pd.to_numeric(successes, errors="coerce").fillna(0).astype(float)
    n = pd.to_numeric(trials, errors="coerce").fillna(0).astype(float)
    out = pd.Series(0.0, index=n.index, dtype=float)
    valid = n > 0
    p = k[valid] / n[valid]
    z = WILSON_Z_ONE_SIDED_95
    out.loc[valid] = (
        p
        + z * z / (2 * n[valid])
        - z * np.sqrt(p * (1 - p) / n[valid] + z * z / (4 * n[valid] ** 2))
    ) / (1 + z * z / n[valid])
    return out.clip(lower=0.0, upper=1.0)


def _validate_and_prepare(df: pd.DataFrame, *, source_name: str) -> tuple[pd.DataFrame, int]:
    missing = sorted(set(REQUIRED_FEATURE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {', '.join(missing)}")

    out = df.copy()
    for col in ("decision_id", "actor_wallet", "market_id", "actor_role", "action", "outcome_token"):
        if out[col].isna().any():
            raise ValueError(f"{source_name}.{col} contains null values")
        out[col] = out[col].astype(str)

    out["actor_wallet"] = out["actor_wallet"].str.lower()
    out["counterparty_wallet"] = out["counterparty_wallet"].where(
        out["counterparty_wallet"].isna(), out["counterparty_wallet"].astype(str).str.lower()
    )
    out["market_id"] = out["market_id"].astype(str)
    out["actor_role"] = out["actor_role"].str.lower()
    out["action"] = out["action"].str.upper()
    out["outcome_token"] = out["outcome_token"].str.lower()
    if "order_hash" in out:
        out["order_hash"] = out["order_hash"].where(
            out["order_hash"].isna(), out["order_hash"].astype(str).str.lower()
        )
        out.loc[out["order_hash"].isin({"", "none", "nan", "<na>"}), "order_hash"] = None

    for col in ("timestamp", "block_number", "log_index"):
        values = pd.to_numeric(out[col], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(f"{source_name}.{col} must contain finite integers")
        if (values % 1 != 0).any():
            raise ValueError(f"{source_name}.{col} must contain integers")
        out[col] = values.astype("int64")

    shares = pd.to_numeric(out["shares"], errors="coerce")
    if shares.isna().any() or (~np.isfinite(shares)).any() or (shares < 0).any():
        raise ValueError(f"{source_name}.shares must contain finite non-negative values")
    out["shares"] = shares.astype(float)

    invalid_roles = sorted(set(out["actor_role"]) - {"maker", "taker"})
    invalid_actions = sorted(set(out["action"]) - {"BUY", "SELL"})
    invalid_tokens = sorted(set(out["outcome_token"]) - {"token1", "token2"})
    if invalid_roles:
        raise ValueError(f"{source_name} has invalid actor_role values: {invalid_roles}")
    if invalid_actions:
        raise ValueError(f"{source_name} has invalid action values: {invalid_actions}")
    if invalid_tokens:
        raise ValueError(f"{source_name} has invalid outcome_token values: {invalid_tokens}")

    before = len(out)
    # Normalize first, so two rows differing only by address casing are exact duplicates.
    out = out.drop_duplicates().copy()
    duplicate_count = before - len(out)
    duplicated_ids = out["decision_id"].duplicated(keep=False)
    if duplicated_ids.any():
        examples = sorted(out.loc[duplicated_ids, "decision_id"].unique())[:5]
        raise ValueError(
            f"{source_name} contains conflicting rows for decision_id(s): {examples}"
        )

    out = out.sort_values(
        ["timestamp", "block_number", "log_index", "decision_id"], kind="stable"
    ).reset_index(drop=True)
    return out, duplicate_count


def build_actor_market_features(decisions: pd.DataFrame) -> pd.DataFrame:
    """Build wallet-market features from the explicitly allowed execution fields."""
    selected = REQUIRED_FEATURE_COLUMNS + [c for c in OPTIONAL_FEATURE_COLUMNS if c in decisions]
    d = decisions[selected].copy()
    d["_day"] = d["timestamp"] // 86_400
    d["_hour"] = d["timestamp"] // 3_600
    d["_is_maker"] = d["actor_role"].eq("maker")
    d["_economic_plus"] = (
        (d["action"].eq("BUY") & d["outcome_token"].eq("token1"))
        | (d["action"].eq("SELL") & d["outcome_token"].eq("token2"))
    )
    keys = ["actor_wallet", "market_id"]
    grouped = d.groupby(keys, sort=True, observed=True)
    features = grouped.agg(
        n_decisions=("decision_id", "size"),
        active_blocks=("block_number", "nunique"),
        active_days=("_day", "nunique"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )

    maker = d[d["_is_maker"]].copy()
    if len(maker):
        if "order_hash" in maker:
            maker["_has_order_hash"] = maker["order_hash"].notna()
            maker["_maker_activity_unit"] = np.where(
                maker["_has_order_hash"],
                "order:" + maker["order_hash"].fillna("").astype(str),
                "block:" + maker["block_number"].astype(str),
            )
        else:
            maker["_has_order_hash"] = False
            maker["_maker_activity_unit"] = "block:" + maker["block_number"].astype(str)
        maker["_plus_volume"] = maker["shares"].where(maker["_economic_plus"], 0.0)
        maker["_minus_volume"] = maker["shares"].where(~maker["_economic_plus"], 0.0)
        maker_grouped = maker.groupby(keys, sort=True, observed=True)
        maker_features = maker_grouped.agg(
            maker_decisions=("decision_id", "size"),
            maker_blocks=("block_number", "nunique"),
            maker_activity_units=("_maker_activity_unit", "nunique"),
            maker_active_days=("_day", "nunique"),
            maker_decisions_with_order_hash=("_has_order_hash", "sum"),
            unique_maker_counterparties=("counterparty_wallet", "nunique"),
            maker_plus_volume=("_plus_volume", "sum"),
            maker_minus_volume=("_minus_volume", "sum"),
        )
        plus_blocks = (
            maker[maker["_economic_plus"]]
            .groupby(keys, sort=True, observed=True)["block_number"]
            .nunique()
            .rename("maker_plus_blocks")
        )
        minus_blocks = (
            maker[~maker["_economic_plus"]]
            .groupby(keys, sort=True, observed=True)["block_number"]
            .nunique()
            .rename("maker_minus_blocks")
        )
        plus_units = (
            maker[maker["_economic_plus"]]
            .groupby(keys, sort=True, observed=True)["_maker_activity_unit"]
            .nunique()
            .rename("maker_plus_activity_units")
        )
        minus_units = (
            maker[~maker["_economic_plus"]]
            .groupby(keys, sort=True, observed=True)["_maker_activity_unit"]
            .nunique()
            .rename("maker_minus_activity_units")
        )
        features = (
            features.join(maker_features)
            .join(plus_blocks)
            .join(minus_blocks)
            .join(plus_units)
            .join(minus_units)
        )

    taker = d[~d["_is_maker"]].copy()
    if len(taker):
        taker_grouped = taker.groupby(keys, sort=True, observed=True)
        taker_features = taker_grouped.agg(
            taker_decisions=("decision_id", "size"),
            taker_blocks=("block_number", "nunique"),
            taker_active_days=("_day", "nunique"),
        )

        # Timing uses one observation per chain block. Without order hashes, this is
        # more resistant to partial fills than a row-level inter-arrival measure.
        taker_blocks = (
            taker.groupby(keys + ["block_number"], sort=True, observed=True)
            .agg(timestamp=("timestamp", "min"), day=("_day", "first"), hour=("_hour", "first"))
            .reset_index()
            .sort_values(keys + ["timestamp", "block_number"], kind="stable")
        )
        taker_blocks["_gap"] = taker_blocks.groupby(keys, sort=False)["timestamp"].diff()
        taker_blocks["_rapid5"] = taker_blocks["_gap"].le(5) & taker_blocks["_gap"].notna()
        gap_features = taker_blocks.groupby(keys, sort=True, observed=True).agg(
            taker_gap_count=("_gap", "count"),
            taker_rapid5_count=("_rapid5", "sum"),
            taker_median_block_gap_seconds=("_gap", "median"),
        )
        hour_counts = (
            taker_blocks.groupby(keys + ["hour"], sort=True, observed=True)
            .size()
            .groupby(keys, sort=True)
            .max()
            .rename("taker_max_blocks_hour")
        )
        day_counts = taker_blocks.groupby(keys + ["day"], sort=True, observed=True).agg(
            blocks=("block_number", "size"), hours=("hour", "nunique")
        )
        day_features = day_counts.groupby(keys, sort=True).agg(
            taker_max_blocks_day=("blocks", "max"),
            taker_max_active_hours_day=("hours", "max"),
        )
        days_12h = (
            day_counts["hours"]
            .ge(12)
            .groupby(keys, sort=True)
            .sum()
            .rename("taker_days_12plus_hours")
        )
        features = (
            features.join(taker_features)
            .join(gap_features)
            .join(hour_counts)
            .join(day_features)
            .join(days_12h)
        )

    count_cols = [
        "maker_decisions",
        "maker_blocks",
        "maker_activity_units",
        "maker_active_days",
        "maker_decisions_with_order_hash",
        "unique_maker_counterparties",
        "maker_plus_blocks",
        "maker_minus_blocks",
        "maker_plus_activity_units",
        "maker_minus_activity_units",
        "taker_decisions",
        "taker_blocks",
        "taker_active_days",
        "taker_gap_count",
        "taker_rapid5_count",
        "taker_max_blocks_hour",
        "taker_max_blocks_day",
        "taker_max_active_hours_day",
        "taker_days_12plus_hours",
    ]
    volume_cols = ["maker_plus_volume", "maker_minus_volume"]
    for col in count_cols:
        if col not in features:
            features[col] = 0
        features[col] = features[col].fillna(0).astype("int64")
    for col in volume_cols:
        if col not in features:
            features[col] = 0.0
        features[col] = features[col].fillna(0.0).astype(float)
    if "taker_median_block_gap_seconds" not in features:
        features["taker_median_block_gap_seconds"] = np.nan

    features["maker_share"] = features["maker_decisions"] / features["n_decisions"]
    features["maker_share_lcb95"] = wilson_lower_bound(
        features["maker_decisions"], features["n_decisions"]
    )
    features["maker_order_hash_coverage"] = np.where(
        features["maker_decisions"] > 0,
        features["maker_decisions_with_order_hash"] / features["maker_decisions"],
        0.0,
    )
    features["maker_activity_unit_mode"] = np.select(
        [
            features["maker_decisions"].eq(0),
            features["maker_order_hash_coverage"].eq(1.0),
            features["maker_order_hash_coverage"].gt(0.0),
        ],
        ["no_maker_rows", "order_hash", "mixed_order_hash_block_fallback"],
        default="distinct_block_fallback",
    )
    maker_total_volume = features["maker_plus_volume"] + features["maker_minus_volume"]
    features["maker_flow_balance"] = np.where(
        maker_total_volume > 0,
        2 * np.minimum(features["maker_plus_volume"], features["maker_minus_volume"])
        / maker_total_volume,
        0.0,
    )
    features["taker_blocks_per_active_day"] = np.where(
        features["taker_active_days"] > 0,
        features["taker_blocks"] / features["taker_active_days"],
        0.0,
    )
    features["taker_rapid5_share"] = np.where(
        features["taker_gap_count"] > 0,
        features["taker_rapid5_count"] / features["taker_gap_count"],
        0.0,
    )
    features["taker_rapid5_lcb95"] = wilson_lower_bound(
        features["taker_rapid5_count"], features["taker_gap_count"]
    )
    return features.reset_index().sort_values(keys, kind="stable").reset_index(drop=True)


def _quantile_higher(values: pd.Series, q: float, floor: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    empirical = float(clean.quantile(q, interpolation="higher")) if len(clean) else float("nan")
    return float(max(floor, empirical)) if np.isfinite(empirical) else float(floor)


def compute_thresholds(reference_features: pd.DataFrame) -> dict[str, Any]:
    """Compute deterministic wallet-weighted peer thresholds by market."""
    maker_global = reference_features[reference_features["maker_activity_units"] >= 20]
    taker_global = reference_features[reference_features["taker_blocks"] >= 20]

    def maker_values(peer: pd.DataFrame) -> dict[str, float]:
        return {
            "maker_activity_units": _quantile_higher(
                peer["maker_activity_units"], 0.90, 150
            ),
            "unique_maker_counterparties": _quantile_higher(
                peer["unique_maker_counterparties"], 0.75, 50
            ),
        }

    def taker_values(peer: pd.DataFrame) -> dict[str, float]:
        return {
            "taker_blocks_per_active_day": _quantile_higher(
                peer["taker_blocks_per_active_day"], 0.95, 30
            ),
            "taker_rapid5_lcb95": _quantile_higher(peer["taker_rapid5_lcb95"], 0.95, 0.15),
            "taker_max_blocks_hour": _quantile_higher(
                peer["taker_max_blocks_hour"], 0.95, 50
            ),
            "taker_max_blocks_day": _quantile_higher(peer["taker_max_blocks_day"], 0.95, 100),
        }

    markets: dict[str, Any] = {}
    for market_id in sorted(reference_features["market_id"].astype(str).unique()):
        market = reference_features[reference_features["market_id"].eq(market_id)]
        maker_local = market[market["maker_activity_units"] >= 20]
        taker_local = market[market["taker_blocks"] >= 20]
        maker_peer = maker_local if len(maker_local) >= 100 else maker_global
        taker_peer = taker_local if len(taker_local) >= 100 else taker_global
        markets[market_id] = {
            "maker": {
                **maker_values(maker_peer),
                "local_peer_count": int(len(maker_local)),
                "calibration_peer_count": int(len(maker_peer)),
                "used_global_fallback": bool(len(maker_local) < 100),
            },
            "taker": {
                **taker_values(taker_peer),
                "local_peer_count": int(len(taker_local)),
                "calibration_peer_count": int(len(taker_peer)),
                "used_global_fallback": bool(len(taker_local) < 100),
            },
        }
    return {
        "method": "wallet-market empirical quantiles with interpolation='higher' and absolute floors",
        "reference_actor_markets": int(len(reference_features)),
        "reference_maker_order_hash_coverage": float(
            reference_features["maker_decisions_with_order_hash"].sum()
            / max(1, reference_features["maker_decisions"].sum())
        ),
        "reference_maker_activity_unit_modes": {
            str(k): int(v)
            for k, v in reference_features["maker_activity_unit_mode"]
            .value_counts()
            .sort_index()
            .items()
        },
        "global": {"maker": maker_values(maker_global), "taker": taker_values(taker_global)},
        "markets": markets,
        "fixed_rules": {
            "market_maker": {
                "maker_active_days_gte": 5,
                "maker_share_wilson_lcb95_gte": 0.85,
                "maker_flow_balance_gte": 0.70,
                "maker_plus_activity_units_gte": 20,
                "maker_minus_activity_units_gte": 20,
            },
            "high_frequency_taker": {
                "taker_blocks_gte": 200,
                "taker_active_days_gte": 3,
                "minimum_timing_signals": 2,
                "round_the_clock_signal_days_12plus_hours_gte": 2,
            },
            "insufficient_history_distinct_blocks_lt": 20,
        },
    }


def _market_thresholds(thresholds: dict[str, Any], market_id: str) -> tuple[dict, dict]:
    market = thresholds["markets"].get(str(market_id))
    if market is None:
        return thresholds["global"]["maker"], thresholds["global"]["taker"]
    return market["maker"], market["taker"]


def classify_actor_markets(features: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    out = features.copy()
    threshold_rows = []
    for market_id in out["market_id"].astype(str):
        maker, taker = _market_thresholds(thresholds, market_id)
        threshold_rows.append(
            {
                "threshold_maker_activity_units": maker["maker_activity_units"],
                "threshold_maker_counterparties": maker["unique_maker_counterparties"],
                "threshold_taker_blocks_per_day": taker["taker_blocks_per_active_day"],
                "threshold_taker_rapid5_lcb95": taker["taker_rapid5_lcb95"],
                "threshold_taker_max_blocks_hour": taker["taker_max_blocks_hour"],
                "threshold_taker_max_blocks_day": taker["taker_max_blocks_day"],
            }
        )
    out = pd.concat([out, pd.DataFrame(threshold_rows, index=out.index)], axis=1)

    side_units_ok = out["maker_plus_activity_units"].ge(20) & out[
        "maker_minus_activity_units"
    ].ge(20)
    mm_support_count = (
        out["maker_flow_balance"].ge(0.70).astype(int)
        + side_units_ok.astype(int)
        + out["unique_maker_counterparties"].ge(out["threshold_maker_counterparties"]).astype(int)
    )
    mm_core = (
        out["maker_active_days"].ge(5)
        & out["maker_activity_units"].ge(out["threshold_maker_activity_units"])
        & out["maker_share_lcb95"].ge(0.85)
    )
    out["strong_market_maker_footprint"] = mm_core & mm_support_count.eq(3)
    out["review_market_maker_footprint"] = (
        ~out["strong_market_maker_footprint"]
        & out["maker_active_days"].ge(5)
        & out["maker_activity_units"].ge(out["threshold_maker_activity_units"])
        & out["maker_share_lcb95"].ge(0.75)
        & mm_support_count.ge(2)
    )

    out["signal_taker_rate"] = out["taker_blocks_per_active_day"].ge(
        out["threshold_taker_blocks_per_day"]
    )
    out["signal_taker_rapid5"] = out["taker_rapid5_lcb95"].ge(
        out["threshold_taker_rapid5_lcb95"]
    )
    out["signal_taker_hour_burst"] = out["taker_max_blocks_hour"].ge(
        out["threshold_taker_max_blocks_hour"]
    )
    out["signal_taker_day_burst"] = out["taker_max_blocks_day"].ge(
        out["threshold_taker_max_blocks_day"]
    )
    out["signal_taker_round_the_clock"] = out["taker_days_12plus_hours"].ge(2)
    signal_cols = [
        "signal_taker_rate",
        "signal_taker_rapid5",
        "signal_taker_hour_burst",
        "signal_taker_day_burst",
        "signal_taker_round_the_clock",
    ]
    out["taker_timing_signal_count"] = out[signal_cols].sum(axis=1).astype("int64")
    hft_eligible = out["taker_blocks"].ge(200) & out["taker_active_days"].ge(3)
    out["strong_high_frequency_taker_footprint"] = hft_eligible & out[
        "taker_timing_signal_count"
    ].ge(2)
    out["review_high_frequency_taker_footprint"] = (
        hft_eligible
        & ~out["strong_high_frequency_taker_footprint"]
        & out["taker_timing_signal_count"].eq(1)
    )
    out["strong_behavioral_exclusion"] = (
        out["strong_market_maker_footprint"]
        | out["strong_high_frequency_taker_footprint"]
    )
    out["review_behavioral_signal"] = (
        out["review_market_maker_footprint"]
        | out["review_high_frequency_taker_footprint"]
    ) & ~out["strong_behavioral_exclusion"]
    out["actor_market_classification"] = np.select(
        [
            out["strong_behavioral_exclusion"],
            out["review_behavioral_signal"],
            out["active_blocks"].lt(20),
        ],
        [LABEL_LIKELY, LABEL_REVIEW, LABEL_INSUFFICIENT],
        default=LABEL_HUMAN,
    )

    def reasons(row: pd.Series) -> str:
        reason = []
        if row["strong_market_maker_footprint"]:
            reason.append("strong_market_maker_footprint")
        if row["strong_high_frequency_taker_footprint"]:
            reason.append("strong_high_frequency_taker_footprint")
        if row["review_market_maker_footprint"]:
            reason.append("review_market_maker_footprint")
        if row["review_high_frequency_taker_footprint"]:
            reason.append("review_high_frequency_taker_footprint")
        if row["active_blocks"] < 20:
            reason.append("fewer_than_20_distinct_blocks_in_market")
        return ";".join(reason) or "no_strong_automation_evidence"

    out["actor_market_reason_codes"] = out.apply(reasons, axis=1)
    return out


def build_activity_screen(reference_decisions: pd.DataFrame) -> pd.DataFrame:
    d = reference_decisions[PROFILE_COLUMNS].copy()
    d["activity_day_utc"] = d["timestamp"] // 86_400
    total = d.groupby("actor_wallet", sort=True)["decision_id"].nunique().rename(
        "activity_total_decisions"
    )
    peak = (
        d.groupby(["actor_wallet", "activity_day_utc"], sort=True)["decision_id"]
        .nunique()
        .groupby("actor_wallet", sort=True)
        .max()
        .rename("activity_peak_utc_day_decisions")
    )
    return pd.concat([total, peak], axis=1).reset_index()


def _load_overrides(path: Path | None) -> pd.DataFrame:
    columns = ["actor_wallet", "override_label", "override_reason", "override_source"]
    if path is None:
        return pd.DataFrame(columns=columns)
    raw = pd.read_csv(path)
    wallet_col = "actor_wallet" if "actor_wallet" in raw else "wallet" if "wallet" in raw else None
    label_col = (
        "override_label"
        if "override_label" in raw
        else "classification"
        if "classification" in raw
        else "label"
        if "label" in raw
        else None
    )
    if wallet_col is None or label_col is None:
        raise ValueError(
            "overrides CSV needs actor_wallet (or wallet) and override_label "
            "(or classification/label)"
        )
    out = pd.DataFrame(
        {
            "actor_wallet": raw[wallet_col].astype(str).str.lower(),
            "override_label": raw[label_col].astype(str).str.lower(),
            "override_reason": raw.get("reason", pd.Series("", index=raw.index)).fillna("").astype(str),
            "override_source": raw.get("source", pd.Series("manual_review", index=raw.index))
            .fillna("manual_review")
            .astype(str),
        }
    )
    aliases = {"exclude": LABEL_LIKELY, "include": LABEL_HUMAN}
    out["override_label"] = out["override_label"].replace(aliases)
    invalid = sorted(set(out["override_label"]) - VALID_LABELS)
    if invalid:
        raise ValueError(f"overrides CSV has unsupported labels: {invalid}")
    if out["actor_wallet"].duplicated().any():
        duplicated = sorted(out.loc[out["actor_wallet"].duplicated(False), "actor_wallet"].unique())
        raise ValueError(f"overrides CSV repeats wallets: {duplicated[:5]}")
    return out[columns]


def build_wallet_features(
    decisions: pd.DataFrame,
    actor_market: pd.DataFrame,
    activity: pd.DataFrame,
    *,
    profile: str,
    include_literature_exclusions: bool,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    d = decisions[REQUIRED_FEATURE_COLUMNS].copy()
    d["_day"] = d["timestamp"] // 86_400
    wallet = d.groupby("actor_wallet", sort=True).agg(
        observed_decisions=("decision_id", "size"),
        observed_distinct_blocks=("block_number", "nunique"),
        observed_active_days=("_day", "nunique"),
        observed_markets=("market_id", "nunique"),
    )
    am = actor_market.groupby("actor_wallet", sort=True).agg(
        strong_market_maker_any=("strong_market_maker_footprint", "max"),
        strong_high_frequency_taker_any=("strong_high_frequency_taker_footprint", "max"),
        review_behavioral_any=("review_behavioral_signal", "max"),
    )
    strong_markets = (
        actor_market[actor_market["strong_behavioral_exclusion"]]
        .groupby("actor_wallet", sort=True)["market_id"]
        .agg(lambda x: ",".join(sorted(set(map(str, x)))))
        .rename("strong_trigger_markets")
    )
    review_markets = (
        actor_market[actor_market["review_behavioral_signal"]]
        .groupby("actor_wallet", sort=True)["market_id"]
        .agg(lambda x: ",".join(sorted(set(map(str, x)))))
        .rename("review_trigger_markets")
    )
    wallet = wallet.join(am).join(strong_markets).join(review_markets).reset_index()
    wallet = wallet.merge(activity, on="actor_wallet", how="left", validate="one_to_one")
    # If an external reference omitted an input wallet, fall back to the observed
    # input history and make the coverage gap explicit.
    wallet["activity_reference_missing_wallet"] = wallet["activity_total_decisions"].isna()
    wallet["activity_total_decisions"] = wallet["activity_total_decisions"].fillna(
        wallet["observed_decisions"]
    ).astype("int64")
    input_peak = (
        d.groupby(["actor_wallet", "_day"], sort=True)["decision_id"]
        .size()
        .groupby("actor_wallet", sort=True)
        .max()
    )
    wallet["activity_peak_utc_day_decisions"] = wallet[
        "activity_peak_utc_day_decisions"
    ].fillna(wallet["actor_wallet"].map(input_peak)).astype("int64")

    profile_rule = ACTIVITY_PROFILES[profile]
    wallet["published_activity_screen"] = (
        wallet["activity_peak_utc_day_decisions"].gt(profile_rule["peak_utc_day_gt"])
        | wallet["activity_total_decisions"].gt(profile_rule["total_decisions_gt"])
    )
    wallet["literature_identified_systematic_arbitrageur"] = (
        wallet["actor_wallet"].isin(LITERATURE_SYSTEMATIC_ARBITRAGE_WALLETS)
        if include_literature_exclusions
        else False
    )
    wallet["strong_behavioral_any"] = (
        wallet["strong_market_maker_any"] | wallet["strong_high_frequency_taker_any"]
    )
    wallet["strong_exclusion_any"] = (
        wallet["strong_behavioral_any"]
        | wallet["published_activity_screen"]
        | wallet["literature_identified_systematic_arbitrageur"]
    )
    wallet["wallet_classification"] = np.select(
        [
            wallet["strong_exclusion_any"],
            wallet["review_behavioral_any"],
            wallet["observed_distinct_blocks"].lt(20),
        ],
        [LABEL_LIKELY, LABEL_REVIEW, LABEL_INSUFFICIENT],
        default=LABEL_HUMAN,
    )

    def reason_codes(row: pd.Series) -> str:
        reason = []
        if row["strong_market_maker_any"]:
            reason.append("strong_market_maker_footprint")
        if row["strong_high_frequency_taker_any"]:
            reason.append("strong_high_frequency_taker_footprint")
        if row["published_activity_screen"]:
            reason.append(f"published_activity_screen_{profile}")
        if row["literature_identified_systematic_arbitrageur"]:
            reason.append("literature_identified_systematic_arbitrageur")
        if row["review_behavioral_any"] and not row["strong_exclusion_any"]:
            reason.append("behavioral_review_signal")
        if row["observed_distinct_blocks"] < 20 and not reason:
            reason.append("fewer_than_20_distinct_blocks")
        return ";".join(reason) or "no_strong_automation_evidence"

    wallet["wallet_reason_codes"] = wallet.apply(reason_codes, axis=1)
    wallet["override_label"] = None
    wallet["override_reason"] = None
    wallet["override_source"] = None
    if len(overrides):
        wallet = wallet.merge(overrides, on="actor_wallet", how="left", suffixes=("", "_new"))
        for col in ("override_label", "override_reason", "override_source"):
            new = f"{col}_new"
            if new in wallet:
                wallet[col] = wallet[new]
                wallet = wallet.drop(columns=new)
        has_override = wallet["override_label"].notna()
        wallet.loc[has_override, "wallet_classification"] = wallet.loc[
            has_override, "override_label"
        ]
        wallet.loc[has_override, "wallet_reason_codes"] = (
            "manual_override:"
            + wallet.loc[has_override, "override_label"].astype(str)
            + np.where(
                wallet.loc[has_override, "override_reason"].fillna("").astype(str).ne(""),
                ":" + wallet.loc[has_override, "override_reason"].fillna("").astype(str),
                "",
            )
        )
    wallet["excluded_by_wallet_policy"] = wallet["wallet_classification"].eq(LABEL_LIKELY)
    for col in ("strong_trigger_markets", "review_trigger_markets"):
        wallet[col] = wallet[col].fillna("")
    return wallet.sort_values("actor_wallet", kind="stable").reset_index(drop=True)


def attach_row_classification(
    decisions: pd.DataFrame,
    actor_market: pd.DataFrame,
    wallet_features: pd.DataFrame,
    *,
    scope: str,
) -> pd.DataFrame:
    am_cols = [
        "actor_wallet",
        "market_id",
        "actor_market_classification",
        "actor_market_reason_codes",
        "strong_behavioral_exclusion",
    ]
    wallet_cols = [
        "actor_wallet",
        "wallet_classification",
        "wallet_reason_codes",
        "published_activity_screen",
        "literature_identified_systematic_arbitrageur",
        "override_label",
    ]
    out = decisions.merge(actor_market[am_cols], on=["actor_wallet", "market_id"], how="left")
    out = out.merge(wallet_features[wallet_cols], on="actor_wallet", how="left", validate="many_to_one")
    if out[["actor_market_classification", "wallet_classification"]].isna().any().any():
        raise RuntimeError("classification join unexpectedly lost rows")

    if scope == "wallet":
        out["row_classification"] = out["wallet_classification"]
        out["row_reason_codes"] = out["wallet_reason_codes"]
    else:
        out["row_classification"] = out["actor_market_classification"]
        out["row_reason_codes"] = out["actor_market_reason_codes"]
        # Published activity and literature labels are wallet identities rather
        # than market-local behavior, and therefore apply under either scope.
        global_likely = (
            out["published_activity_screen"]
            | out["literature_identified_systematic_arbitrageur"]
        )
        out.loc[global_likely, "row_classification"] = LABEL_LIKELY
        out.loc[global_likely, "row_reason_codes"] = out.loc[
            global_likely, "wallet_reason_codes"
        ]
        # A reviewed wallet override is intentionally global and has final say.
        overridden = out["override_label"].notna()
        out.loc[overridden, "row_classification"] = out.loc[overridden, "override_label"]
        out.loc[overridden, "row_reason_codes"] = out.loc[overridden, "wallet_reason_codes"]

    out["excluded_by_policy"] = out["row_classification"].eq(LABEL_LIKELY)
    out["human_candidate"] = out["row_classification"].eq(LABEL_HUMAN)
    return out.sort_values(
        ["timestamp", "block_number", "log_index", "decision_id"], kind="stable"
    ).reset_index(drop=True)


def classify_decisions(
    decisions: pd.DataFrame,
    *,
    reference_decisions: pd.DataFrame | None = None,
    profile: str = "strict",
    scope: str = "wallet",
    overrides: pd.DataFrame | None = None,
    include_literature_exclusions: bool = True,
) -> dict[str, Any]:
    """Classify a DataFrame and return all tables without writing files."""
    if profile not in ACTIVITY_PROFILES:
        raise ValueError(f"unsupported profile {profile!r}")
    if scope not in {"wallet", "market"}:
        raise ValueError("scope must be 'wallet' or 'market'")
    clean, duplicates_dropped = _validate_and_prepare(decisions, source_name="input")
    if reference_decisions is None:
        reference = clean
        reference_duplicates_dropped = duplicates_dropped
        reference_scope = "input_only_left_censored"
    else:
        reference, reference_duplicates_dropped = _validate_and_prepare(
            reference_decisions, source_name="reference_decisions"
        )
        reference_scope = "external_reference"

    input_features = build_actor_market_features(clean)
    reference_features = build_actor_market_features(reference)
    thresholds = compute_thresholds(reference_features)
    actor_market = classify_actor_markets(input_features, thresholds)
    activity = build_activity_screen(reference)
    override_table = overrides if overrides is not None else _load_overrides(None)
    wallet = build_wallet_features(
        clean,
        actor_market,
        activity,
        profile=profile,
        include_literature_exclusions=include_literature_exclusions,
        overrides=override_table,
    )
    classified = attach_row_classification(clean, actor_market, wallet, scope=scope)
    partitions = {
        "human_candidate": classified[classified["row_classification"].eq(LABEL_HUMAN)].copy(),
        "uncertain": classified[
            classified["row_classification"].isin({LABEL_REVIEW, LABEL_INSUFFICIENT})
        ].copy(),
        "excluded": classified[classified["row_classification"].eq(LABEL_LIKELY)].copy(),
    }
    if sum(len(v) for v in partitions.values()) != len(classified):
        raise RuntimeError("classification partitions are not exhaustive")
    return {
        "classified": classified,
        "actor_market_features": actor_market,
        "wallet_features": wallet,
        "thresholds": thresholds,
        "partitions": partitions,
        "input_duplicates_dropped": duplicates_dropped,
        "reference_duplicates_dropped": reference_duplicates_dropped,
        "reference_scope": reference_scope,
    }


def run_to_directory(
    input_path: Path,
    out_dir: Path,
    *,
    reference_path: Path | None = None,
    profile: str = "strict",
    scope: str = "wallet",
    overrides_path: Path | None = None,
    include_literature_exclusions: bool = True,
) -> dict[str, Any]:
    decisions = pd.read_parquet(input_path)
    reference = pd.read_parquet(reference_path) if reference_path is not None else None
    overrides = _load_overrides(overrides_path)
    result = classify_decisions(
        decisions,
        reference_decisions=reference,
        profile=profile,
        scope=scope,
        overrides=overrides,
        include_literature_exclusions=include_literature_exclusions,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "classified_decisions": out_dir / "classified_decisions.parquet",
        "human_candidate_decisions": out_dir / "human_candidate_decisions.parquet",
        "uncertain_decisions": out_dir / "uncertain_decisions.parquet",
        "excluded_decisions": out_dir / "excluded_decisions.parquet",
        "actor_market_features": out_dir / "actor_market_features.parquet",
        "wallet_features": out_dir / "wallet_features.parquet",
        "wallet_classification_csv": out_dir / "wallet_classification.csv",
        "thresholds": out_dir / "thresholds.json",
    }
    result["classified"].to_parquet(output_paths["classified_decisions"], index=False)
    result["partitions"]["human_candidate"].to_parquet(
        output_paths["human_candidate_decisions"], index=False
    )
    result["partitions"]["uncertain"].to_parquet(
        output_paths["uncertain_decisions"], index=False
    )
    result["partitions"]["excluded"].to_parquet(
        output_paths["excluded_decisions"], index=False
    )
    result["actor_market_features"].to_parquet(output_paths["actor_market_features"], index=False)
    result["wallet_features"].to_parquet(output_paths["wallet_features"], index=False)
    result["wallet_features"].to_csv(output_paths["wallet_classification_csv"], index=False)
    _json_dump(result["thresholds"], output_paths["thresholds"])

    labels = result["classified"]["row_classification"].value_counts().sort_index()
    wallet_labels = result["wallet_features"]["wallet_classification"].value_counts().sort_index()
    manifest: dict[str, Any] = {
        "classifier_version": CLASSIFIER_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "rows_read": int(len(decisions)),
            "exact_duplicate_rows_dropped": int(result["input_duplicates_dropped"]),
        },
        "reference": {
            "path": str(reference_path) if reference_path is not None else str(input_path),
            "sha256": sha256_file(reference_path) if reference_path is not None else sha256_file(input_path),
            "scope": result["reference_scope"],
            "left_censored_to_imported_markets": bool(reference_path is None),
            "exact_duplicate_rows_dropped": int(result["reference_duplicates_dropped"]),
            "input_wallets_missing_from_external_reference": int(
                result["wallet_features"]["activity_reference_missing_wallet"].sum()
            ),
        },
        "configuration": {
            "profile": profile,
            "activity_profile_rule": ACTIVITY_PROFILES[profile],
            "scope": scope,
            "literature_exclusions_enabled": include_literature_exclusions,
            "overrides_path": str(overrides_path) if overrides_path is not None else None,
        },
        "sources": {
            "published_activity_screen": DELLA_VEDOVA_SOURCE,
            "literature_wallets": LITERATURE_SOURCE,
        },
        "feature_boundary": REQUIRED_FEATURE_COLUMNS
        + [c for c in OPTIONAL_FEATURE_COLUMNS if c in result["classified"].columns],
        "guardrails": {
            "settlement_payoff_or_correctness_used": False,
            "maker_timestamps_used_as_hft_speed_evidence": False,
            "timing_counts_distinct_blocks": True,
            "maker_order_hash_available": bool("order_hash" in result["classified"].columns),
            "maker_order_hash_coverage": float(
                result["actor_market_features"]["maker_decisions_with_order_hash"].sum()
                / max(1, result["actor_market_features"]["maker_decisions"].sum())
            ),
            "maker_activity_unit_modes": {
                str(k): int(v)
                for k, v in result["actor_market_features"]["maker_activity_unit_mode"]
                .value_counts()
                .sort_index()
                .items()
            },
            "fill_fragmentation_note": (
                "Maker activity uses distinct non-null order_hash values when present and collapses "
                "rows without a hash to distinct block fallback units. Taker timing always uses "
                "distinct blocks. The published activity screen counts distinct decision IDs/"
                "participations and can still be inflated by multiple fills of one resting maker order."
            ),
            "human_label_note": (
                "human_candidate means no implemented screen fired with enough observed history; "
                "it is not verified human identity."
            ),
        },
        "counts": {
            "classified_rows": int(len(result["classified"])),
            "row_labels": {str(k): int(v) for k, v in labels.items()},
            "wallet_labels": {str(k): int(v) for k, v in wallet_labels.items()},
            "human_candidate_rows": int(len(result["partitions"]["human_candidate"])),
            "uncertain_rows": int(len(result["partitions"]["uncertain"])),
            "excluded_rows": int(len(result["partitions"]["excluded"])),
        },
        "outputs": {},
    }
    for name, path in output_paths.items():
        manifest["outputs"][name] = {"path": str(path), "sha256": sha256_file(path)}
    manifest_path = out_dir / "classification_manifest.json"
    _json_dump(manifest, manifest_path)
    result["manifest"] = manifest
    result["output_paths"] = {**output_paths, "manifest": manifest_path}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="market_decisions.parquet")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--reference-decisions",
        type=Path,
        default=None,
        help=(
            "broader decisions parquet used to calibrate thresholds and lifetime/peak-day activity; "
            "without it, the imported input is explicitly treated as left-censored"
        ),
    )
    parser.add_argument("--scope", choices=("wallet", "market"), default="wallet")
    parser.add_argument(
        "--profile",
        choices=tuple(ACTIVITY_PROFILES),
        default="strict",
        help="published-activity sensitivity profile; strict is the high-purity default",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help=(
            "reviewed CSV with actor_wallet and label/classification; labels may be exclude, include, "
            "likely_automated_or_mm, review, insufficient_history, or human_candidate"
        ),
    )
    parser.add_argument(
        "--disable-literature-exclusions",
        action="store_true",
        help="do not apply the six Saguillo et al. systematic-arbitrage wallet labels",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_to_directory(
        args.input,
        args.out,
        reference_path=args.reference_decisions,
        profile=args.profile,
        scope=args.scope,
        overrides_path=args.overrides,
        include_literature_exclusions=not args.disable_literature_exclusions,
    )
    counts = result["manifest"]["counts"]
    print(
        f"classified {counts['classified_rows']:,} decisions: "
        f"{counts['human_candidate_rows']:,} human candidates, "
        f"{counts['uncertain_rows']:,} uncertain, {counts['excluded_rows']:,} excluded"
    )
    print(f"outputs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
