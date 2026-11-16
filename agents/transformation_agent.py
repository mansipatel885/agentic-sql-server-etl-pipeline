"""
Transformation Agent.

Step 1 (run):   decide, from the mapping and the two schemas, which cleaning /
                conversion steps this job needs, and record them as a plan.
Step 2 (apply): the Execution Agent calls this on the extracted rows. It renames
                columns, trims text, converts types, then checks EVERY row
                against the destination (NOT NULL, maximum text length).

Safety rule: rows that cannot be loaded make the run FAIL before anything is
written (default `on_bad_rows: fail`). The watermark is not advanced, and the
bad rows are saved to a CSV so they can be fixed. `on_bad_rows: quarantine`
loads the good rows and saves the bad ones, but then those rows are only in
the CSV, so use it knowingly.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd

from .base import LOG_DIR, Agent, AgentResult, DataQualityError, RunContext

log = logging.getLogger(__name__)


class TransformationAgent(Agent):
    name = "Transformation Agent"

    # ------------------------------------------------------------------ plan
    def _run(self, ctx: RunContext) -> AgentResult:
        cfg = ctx.job_cfg
        steps = []
        renamed = ctx.stats.get("renamed_columns", {})
        if renamed:
            steps.append("rename columns: " + ", ".join(f"{s} -> {d}" for s, d in renamed.items()))
        if cfg.get("exclude_columns"):
            steps.append("drop excluded columns: " + ", ".join(cfg["exclude_columns"]))
        if cfg.get("trim_strings", True):
            n_text = sum(1 for d in ctx.mapping.values() if ctx.dest_schema[d].category == "string")
            if n_text:
                steps.append(f"trim leading/trailing spaces in {n_text} text column(s)")
        for s, (a, b) in ctx.stats.get("conversions", {}).items():
            steps.append(f"convert {s}: {a} -> {b}")
        steps.append("check every row against the destination (NOT NULL columns, maximum text length)")
        ctx.transform_steps = steps
        return AgentResult(self.name, True, f"{len(steps)} step(s) planned")

    # ----------------------------------------------------------------- apply
    def apply(self, ctx: RunContext, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Return (clean rows in destination column names, number of rejected rows)."""
        cfg = ctx.job_cfg
        trim = cfg.get("trim_strings", True)
        policy = cfg.get("on_bad_rows", "fail")

        keep = [c for c in df.columns if c in ctx.mapping]
        out = df[keep].rename(columns=ctx.mapping).copy()
        bad = pd.Series(False, index=out.index)
        reasons: Dict[str, int] = {}

        def flag(mask: pd.Series, why: str) -> None:
            nonlocal bad
            n = int(mask.sum())
            if n:
                reasons[why] = reasons.get(why, 0) + n
                bad = bad | mask

        for src, dst in ctx.mapping.items():
            if dst not in out.columns:
                continue
            info, src_info = ctx.dest_schema[dst], ctx.source_schema[src]
            series = out[dst]

            if trim and info.category == "string":
                series = series.map(lambda v: v.strip() if isinstance(v, str) else v)
            if src_info.category != info.category:
                series, conv_bad = self._convert(series, info.category)
                flag(conv_bad, f"{dst}: value cannot be converted to {info.category}")
            if not info.nullable and not info.has_default and not info.autoincrement:
                flag(series.isna(), f"{dst}: NULL in a NOT NULL column")
            if info.category == "string" and info.length:
                flag(series.notna() & (series.astype(str).str.len() > info.length), f"{dst}: text longer than {info.length}")
            out[dst] = series

        n_bad = int(bad.sum())
        ctx.stats["rows_rejected"] = n_bad
        if n_bad == 0:
            return out, 0

        path = LOG_DIR / f"rejected_{ctx.job_name}_{ctx.run_id}.csv"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        df.loc[bad[bad].index].to_csv(path, index=False)
        detail = "; ".join(f"{why} ({count})" for why, count in reasons.items())
        ctx.stats["rejected_file"] = str(path)

        if policy == "fail":
            raise DataQualityError(f"{n_bad} of {len(out)} rows cannot be loaded: {detail}. Rejected rows saved to {path}")
        log.warning("%d rows quarantined to %s (%s)", n_bad, path, detail)
        return out.loc[~bad], n_bad

    @staticmethod
    def _convert(series: pd.Series, category: str) -> Tuple[pd.Series, pd.Series]:
        """Convert a column to `category`; returns (converted, mask of values that failed)."""
        present = series.notna()
        if category == "integer":
            num = pd.to_numeric(series, errors="coerce")
            non_int = num.notna() & (num % 1 != 0)
            failed = (num.isna() & present) | non_int
            return num.where(~non_int).astype("Int64"), failed
        if category == "decimal":
            num = pd.to_numeric(series, errors="coerce")
            return num, num.isna() & present
        if category in ("datetime", "date"):
            parsed = pd.to_datetime(series, errors="coerce")
            failed = parsed.isna() & present
            return (parsed.dt.date if category == "date" else parsed), failed
        if category == "string":
            return series.where(~present, series.astype(str)), pd.Series(False, index=series.index)
        return series, pd.Series(False, index=series.index)
