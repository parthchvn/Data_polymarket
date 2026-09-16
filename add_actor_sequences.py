"""Add actor_id and sequence columns to an existing data/scored/ directory without rerunning the pipeline.
Usage: python add_actor_sequences.py [--out data/scored]"""
import argparse, gzip, json
from pathlib import Path
import pandas as pd
from xvi_score import add_actor_sequences, actor_id_of, write_trajectories, DECISION_COLS

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/scored"); a = ap.parse_args(); out = Path(a.out)
d = pd.read_parquet(out / "market_decisions.parquet")
d = add_actor_sequences(d.drop(columns=[c for c in ("actor_id", "actor_seq", "actor_market_seq", "actor_n_decisions", "prev_decision_timestamp") if c in d]))
cols = [c for c in DECISION_COLS if c in d]
d[cols].to_parquet(out / "market_decisions.parquet", index=False)
for name in ("actor_summary.parquet", "actor_summary_all_markets.parquet"):
    s = pd.read_parquet(out / name)
    if "actor_id" in s: s = s.drop(columns=["actor_id"])
    s.insert(0, "actor_id", s["actor_wallet"].map(actor_id_of).astype("int64")); s.to_parquet(out / name, index=False)
write_trajectories(d[cols], out / "actor_trajectories.jsonl.gz")
print(f"{len(d):,} decisions, {d.actor_id.nunique():,} actors; wrote market_decisions.parquet, actor_summary*.parquet, actor_trajectories.jsonl.gz")
