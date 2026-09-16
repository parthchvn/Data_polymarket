# XVI settlement-payoff pipeline (Polymarket, Wang dataset)

Workflow: select market -> extract raw fills -> verify settlement on chain -> score each decision
independently -> sum by actor -> inspect and export.

## Steps and commands

```
pip install pyarrow pandas requests pycryptodome pytest

# 1. row-group index over trades.parquet (once per dataset revision, 31 MB read, ~40 s)
python xvi_extract.py index

# 2. Wang's processed maker rows for the markets (used for linkage checks and transaction hashes)
python xvi_extract.py lookup  --markets 507300 2446852 570360 542539
python xvi_extract.py extract --markets 507300 2446852 570360 542539 --out data/selected

# 3. raw OrderFilled events for those markets' tokens, maker AND taker orders, from orderfilled.parquet
#    (column-projected, month-partitioned; 14.6 GB for these four markets)
python xvi_extract.py fills

# 4. settlement payouts + resolution timestamps from the ConditionalTokens contract on Polygon
#    (set POLYGON_RPC to an archive endpoint if the default free one rate-limits)
python xvi_settlement.py

# 5. decisions, scores, actor sums, exports
python xvi_score.py run
python xvi_score.py wallet --wallet 0x... --market 507300

# 6. conservative candidate-human screen (wallet-wide by default)
python filter_human_trades.py \
  --input data/scored/market_decisions.parquet \
  --out data/human_filter \
  --profile strict

# Prefer venue-wide decision history for wallet activity and peer thresholds when available
python filter_human_trades.py \
  --input data/scored/market_decisions.parquet \
  --reference-decisions /path/to/venue_wide_market_decisions.parquet \
  --out data/human_filter \
  --profile strict

python -m pytest tests -q
```

## Outputs (data/scored/)

- `market_decisions.parquet` / `market_decisions.jsonl.gz`: one row per actor decision, sorted by
  (timestamp, block_number, log_index). Key columns: decision_id, actor_wallet, actor_role (maker/taker),
  action, outcome_token, outcome_label, shares, execution_price, cash_amount_usd, fee_amount,
  order_hash (when present in the extract), settlement_payout, settlement_timestamp, label_available_at,
  decision_payoff_usd, decision_payoff_after_fee_usd, score_status, label_public_at_decision,
  fill_group_consistent.
- `actor_summary.parquet`: per (wallet, market) sum of scored decision payoffs, counts, net and minimum
  running positions per token, coverage_status, external_inventory_detected. The label is
  "Total settlement payoff of observed trades", never "realized profit".
- `actor_summary_all_markets.parquet`: same summed over the imported markets only.
- `scoring_manifest.json`: counts, checks, settlement evidence, semantics evidence, checksums.

## Candidate-human filter

`filter_human_trades.py` classifies wallets from execution behavior without using settlement outcomes,
payoff, profitability, or correctness. The default `strict` profile prioritizes sample purity and writes
three disjoint decision files:

- `data/human_filter/human_candidate_decisions.parquet`: sufficient observed history and no implemented
  automation, market-making, or literature screen hit;
- `data/human_filter/uncertain_decisions.parquet`: review signals or fewer than 20 distinct observed blocks;
- `data/human_filter/excluded_decisions.parquet`: likely automated/market-maker behavior or a documented
  systematic-arbitrage wallet.

The directory also contains every classified row, wallet and wallet-market audit features, the calibrated
thresholds, CSV labels, file hashes, and a run manifest. `strict`, `balanced`, and `lenient` activity profiles
implement sensitivity cutoffs of 20/500, 50/1,000, and 100/2,000 (peak UTC-day/lifetime decisions,
respectively). Use `--scope market` only when market-local removal is intended, and `--overrides` for a
reviewed allow/deny CSV.

No fill-only rule can verify biological humanity. `human_candidate` means that no implemented screen fired,
not that the wallet owner was identified. See [HUMAN_FILTER_METHODOLOGY.md](HUMAN_FILTER_METHODOLOGY.md) for
the research basis, exact rules, limitations, and citations.

## Semantics that had to be established (see scoring_manifest.json for the evidence)

- Wang's trades.parquet keeps only maker-order events, mirrors the maker side onto the taker (wrong for
  MINT/MERGE fills), rounds to 2 dp, and drops every SELL on CTF_EXCHANGE_V2. Decisions are therefore
  built from orderfilled.parquet events, one per event, attributed to the event's `maker` (the order owner).
- V1 contracts: the collateral leg is asset id 0. V2: first id field is a side flag (0 BUY / 1 SELL),
  second is the token id. Both confirmed against Polygon receipts and by the fill-group invariant
  (sum of maker fill shares == taker event shares) and by the market-wide zero sum of payoffs.
- Fees (V2 only in this data) are settled in collateral on both sides; decision_payoff_usd is before fees.

## Not verified / limits

- 10,529 V2 taker decisions have no transaction_hash (their maker counterparts are absent from trades.parquet).
- The legacy selected fill artifact has no `order_hash`. New `xvi_extract.py fills` runs retain it so the
  filter can recognize multiple maker fills of one order; legacy scoring remains supported with a distinct-
  block fallback.
- 5 decisions in 507300 (4 fill groups) have counterparties missing from the dataset; flagged.
- Actor sums are sums of observed exchange fills. Transfers, splits, merges and neg-risk conversions are not
  observed; 426 actor-markets show negative running inventory and are flagged.
- No dashboard is included yet.
