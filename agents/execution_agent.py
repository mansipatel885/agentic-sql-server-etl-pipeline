"""
Execution Agent: extract -> transform -> load, with the same watermark safety
rules as the original pipeline.

  * full load:        read the whole source table, load it
  * incremental load: read rows past the stored watermark, load them, and move
                      the watermark forward ONLY after the load succeeded

On any failure the watermark is left untouched (control.mark_run_failed) and the
exception is re-raised so the Error & Alert Agent can deal with it.

The loader, control module and query builder can be swapped (that is how the
tests run without a SQL Server); by default they are the existing project code.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import pandas as pd
from sqlalchemy import text

from .base import Agent, AgentResult, RunContext
from .transformation_agent import TransformationAgent

log = logging.getLogger(__name__)


def default_loader(ctx: RunContext, df: pd.DataFrame) -> int:
    """Write with the project's existing strategies (truncate_load / append / upsert)."""
    from config import DEST_CONN_STR
    from load_utils import write_append, write_truncate_load, write_upsert

    cfg = ctx.job_cfg
    strategy, table = cfg["write_strategy"], cfg["dest_table"]
    if strategy == "truncate_load":
        return write_truncate_load(ctx.dest_engine, DEST_CONN_STR, df, table)
    if strategy == "append":
        return write_append(ctx.dest_engine, df, table)
    if strategy == "upsert":
        return write_upsert(ctx.dest_engine, DEST_CONN_STR, df, table, cfg.get("primary_key"))
    raise ValueError(f"Unsupported write_strategy '{strategy}'")


class ExecutionAgent(Agent):
    name = "Execution Agent"

    def __init__(
        self,
        transformer: Optional[TransformationAgent] = None,
        loader: Optional[Callable[[RunContext, pd.DataFrame], int]] = None,
        control: Any = None,
        query_builder: Optional[Callable[..., Any]] = None,
    ):
        self.transformer = transformer or TransformationAgent()
        self.loader = loader or default_loader
        self._control = control
        self._query_builder = query_builder

    # lazy defaults keep this module importable without pyodbc / a config file
    @property
    def control(self):
        if self._control is None:
            import control

            self._control = control
        return self._control

    @property
    def query_builder(self):
        if self._query_builder is None:
            from incremental_load import _build_query

            self._query_builder = _build_query
        return self._query_builder

    # ------------------------------------------------------------------- run
    def _run(self, ctx: RunContext) -> AgentResult:
        if ctx.dry_run:
            return AgentResult(self.name, True, "dry run: nothing was read or written")
        if ctx.job_cfg["load_type"] == "full":
            return self._full(ctx)
        return self._incremental(ctx)

    def _full(self, ctx: RunContext) -> AgentResult:
        cfg = ctx.job_cfg
        df = pd.read_sql(text(f"SELECT * FROM {cfg['source_table']}"), ctx.source_engine)
        ctx.stats["rows_extracted"] = len(df)
        if df.empty:
            ctx.stats["rows_written"] = 0
            return AgentResult(self.name, True, "source table is empty; nothing loaded")
        clean, _ = self.transformer.apply(ctx, df)
        written = self.loader(ctx, clean)
        ctx.stats["rows_written"] = written
        return AgentResult(self.name, True, f"full load: {len(df)} rows read, {written} written")

    def _incremental(self, ctx: RunContext) -> AgentResult:
        cfg, job = ctx.job_cfg, ctx.job_name
        wm_col, wm_type = cfg["watermark_column"], cfg["watermark_type"]
        control = self.control

        control.ensure_job_registered(job, wm_type)
        if ctx.full_reload:
            log.info("[%s] full reload requested: resetting watermark", job)
            control.reset_watermark(job)
        run_start = control.mark_run_start(job)
        last = control.get_last_watermark(job)
        ctx.stats["watermark_before"] = last

        try:
            query, params = self.query_builder(cfg["source_table"], wm_col, wm_type, last, run_start)
            df = pd.read_sql(text(query), ctx.source_engine, params=params)
            ctx.stats["rows_extracted"] = len(df)

            if df.empty:
                control.mark_run_success(job, last)
                ctx.stats.update(rows_written=0, watermark_after=last)
                return AgentResult(self.name, True, "no new rows; watermark unchanged")

            new_watermark = df[wm_col].max()  # taken from the SOURCE column, before any renaming
            clean, _ = self.transformer.apply(ctx, df)
            written = self.loader(ctx, clean)

            control.mark_run_success(job, new_watermark)  # only now does the bookmark move
            ctx.stats.update(rows_written=written, watermark_after=str(new_watermark))
            return AgentResult(self.name, True, f"incremental load: {len(df)} rows read, {written} written, watermark -> {new_watermark}")
        except Exception:
            control.mark_run_failed(job)  # watermark stays where it was; next run retries
            raise
