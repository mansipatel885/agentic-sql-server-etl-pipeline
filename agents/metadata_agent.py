"""
Metadata Collection Agent.

After every run (success or failure) it records what happened:
who/what/when, row counts, watermark before and after, schema fingerprints,
warning and error counts, per-agent timings and the error class if there was one.

Two places, so the record survives a database outage:
  1. logs/run_metadata.jsonl   (always)
  2. table etl_run_metadata    (in the destination database)

It also compares the source schema fingerprint with the last successful run and
notes it if the source table changed in between.

It never raises: a problem here must not turn a good load into a failed one.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, select

from .base import ERROR, INFO, LOG_DIR, WARNING, Agent, AgentResult, Finding, RunContext
from .schema_utils import schema_fingerprint

log = logging.getLogger(__name__)


def _table() -> Table:
    md = MetaData()
    return Table(
        "etl_run_metadata", md,
        Column("run_id", String(32), primary_key=True),
        Column("job_name", String(200)),
        Column("load_type", String(20)),
        Column("write_strategy", String(20)),
        Column("source_table", String(200)),
        Column("dest_table", String(200)),
        Column("status", String(20)),
        Column("started_at", DateTime),
        Column("finished_at", DateTime),
        Column("duration_s", Float),
        Column("rows_extracted", Integer),
        Column("rows_written", Integer),
        Column("rows_rejected", Integer),
        Column("watermark_before", String(100)),
        Column("watermark_after", String(100)),
        Column("source_fingerprint", String(32)),
        Column("dest_fingerprint", String(32)),
        Column("n_errors", Integer),
        Column("n_warnings", Integer),
        Column("error_stage", String(100)),
        Column("error_class", String(50)),
        Column("error_message", String(1000)),
        Column("agent_timings", String(2000)),
    )


def _naive(dt: Optional[datetime]) -> Optional[datetime]:
    return dt.replace(tzinfo=None) if dt is not None else None


class MetadataAgent(Agent):
    name = "Metadata Agent"

    def run(self, ctx: RunContext) -> AgentResult:  # never raises
        try:
            return super().run(ctx)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("metadata collection failed: %s", exc)
            return AgentResult(self.name, True, "metadata collection failed", [Finding(WARNING, "METADATA_FAILED", str(exc))])

    def build_record(self, ctx: RunContext) -> Dict[str, Any]:
        cfg, stats = ctx.job_cfg, ctx.stats
        findings = ctx.all_findings
        started, finished = ctx.started_at, ctx.finished_at
        duration = (finished - started).total_seconds() if finished and started else None
        message = str(ctx.error) if ctx.error else None
        return {
            "run_id": ctx.run_id,
            "job_name": ctx.job_name,
            "load_type": cfg.get("load_type"),
            "write_strategy": cfg.get("write_strategy"),
            "source_table": cfg.get("source_table"),
            "dest_table": cfg.get("dest_table"),
            "status": ctx.status,
            "started_at": _naive(started),
            "finished_at": _naive(finished),
            "duration_s": round(duration, 3) if duration is not None else None,
            "rows_extracted": stats.get("rows_extracted"),
            "rows_written": stats.get("rows_written"),
            "rows_rejected": stats.get("rows_rejected"),
            "watermark_before": None if stats.get("watermark_before") is None else str(stats["watermark_before"]),
            "watermark_after": None if stats.get("watermark_after") is None else str(stats["watermark_after"]),
            "source_fingerprint": schema_fingerprint(ctx.source_schema) if ctx.source_schema else None,
            "dest_fingerprint": schema_fingerprint(ctx.dest_schema) if ctx.dest_schema else None,
            "n_errors": sum(f.severity == ERROR for f in findings),
            "n_warnings": sum(f.severity == WARNING for f in findings),
            "error_stage": ctx.error_stage,
            "error_class": (ctx.alert or {}).get("error_class"),
            "error_message": message[:1000] if message else None,
            "agent_timings": json.dumps({r.agent: r.duration_s for r in ctx.results})[:2000],
        }

    def _run(self, ctx: RunContext) -> AgentResult:
        if ctx.dry_run:
            return AgentResult(self.name, True, "dry run: no metadata stored")

        record = self.build_record(ctx)
        findings: List[Finding] = []

        # 1. local file: always
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_DIR / "run_metadata.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            findings.append(Finding(WARNING, "METADATA_FILE_FAILED", str(exc)))

        # 2. database table
        stored = "file only"
        if ctx.dest_engine is not None:
            try:
                table = _table()
                table.metadata.create_all(ctx.dest_engine)
                with ctx.dest_engine.begin() as conn:
                    previous = conn.execute(
                        select(table.c.source_fingerprint)
                        .where(table.c.job_name == ctx.job_name, table.c.status == "success")
                        .order_by(table.c.finished_at.desc())
                        .limit(1)
                    ).scalar()
                    if previous and record["source_fingerprint"] and previous != record["source_fingerprint"]:
                        findings.append(Finding(INFO, "SCHEMA_CHANGED_SINCE_LAST_RUN", "the source table's columns or types changed since the last successful run"))
                    conn.execute(table.insert().values(**record))
                stored = "file + database"
            except Exception as exc:
                findings.append(Finding(WARNING, "METADATA_DB_FAILED", f"could not store run metadata in the database: {exc}"))

        return AgentResult(self.name, True, f"run {ctx.run_id} recorded ({stored})", findings)
