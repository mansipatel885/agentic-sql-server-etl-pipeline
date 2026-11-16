"""
The orchestrator: runs the agents in order and hands failures to the
Error and Alert Agent.

    Requirement -> Validation -> Mapping -> Transformation -> Execution
                                                                 |
              (always, success or failure)  Metadata <-----------+
              (on any failure)              Error & Alert

Nothing is written to the destination unless every earlier agent said OK.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from agents import (
    ErrorAlertAgent, ExecutionAgent, MappingAgent, MetadataAgent, RequirementAgent,
    RunContext, StageFailed, TransformationAgent, ValidationAgent,
)
from agents.base import ERROR, WARNING

log = logging.getLogger(__name__)


class AgenticPipeline:
    def __init__(
        self,
        requirement=None, validation=None, mapping=None, transformation=None,
        execution=None, metadata=None, error_alert=None,
        source_engine=None, dest_engine=None,
    ):
        transformation = transformation or TransformationAgent()
        self.stages = [
            requirement or RequirementAgent(),
            validation or ValidationAgent(),
            mapping or MappingAgent(),
            transformation,
            execution or ExecutionAgent(transformation),
        ]
        self.metadata = metadata or MetadataAgent()
        self.error_alert = error_alert or ErrorAlertAgent()
        self._source_engine, self._dest_engine = source_engine, dest_engine

    def _attach_engines(self, ctx: RunContext) -> None:
        if self._source_engine is None or self._dest_engine is None:
            from config import DEST_CONN_STR, SOURCE_CONN_STR
            from db import get_engine

            self._source_engine = self._source_engine or get_engine(SOURCE_CONN_STR)
            self._dest_engine = self._dest_engine or get_engine(DEST_CONN_STR)
        ctx.source_engine, ctx.dest_engine = self._source_engine, self._dest_engine

    def run(self, job_name: Optional[str] = None, request_text: Optional[str] = None,
            full_reload: bool = False, dry_run: bool = False) -> RunContext:
        ctx = RunContext(job_name=job_name, request_text=request_text, full_reload=full_reload, dry_run=dry_run)
        stage_name = "Setup"
        try:
            self._attach_engines(ctx)
            for stage in self.stages:
                stage_name = stage.name
                result = stage.run(ctx)
                ctx.results.append(result)
                log.info("[%s] %s: %s", ctx.job_name, stage.name, result.summary)
                if not result.ok:
                    raise StageFailed(stage.name, result)
            ctx.status = "dry_run_ok" if dry_run else "success"
        except Exception as exc:
            ctx.status, ctx.error = "failed", exc
            ctx.error_stage = getattr(exc, "stage", None) or stage_name
            try:
                ctx.alert = self.error_alert.handle(ctx, exc)
            except Exception:  # pragma: no cover - the alert agent must not hide the real error
                log.exception("error agent failed")
        finally:
            ctx.finished_at = datetime.now(timezone.utc)
            ctx.results.append(self.metadata.run(ctx))
        return ctx


def format_report(ctx: RunContext) -> str:
    """A readable summary of one run, stage by stage."""
    lines: List[str] = [f"Run {ctx.run_id}  job={ctx.job_name}  status={ctx.status.upper()}", ""]
    for r in ctx.results:
        mark = "OK  " if r.ok else "STOP"
        lines.append(f"[{mark}] {r.agent:<22} {r.summary}  ({r.duration_s}s)")
        for f in r.findings:
            if f.severity in (ERROR, WARNING):
                lines.append(f"         {f}")
    if ctx.transform_steps and ctx.dry_run:
        lines += ["", "Transformation plan:"] + [f"  - {s}" for s in ctx.transform_steps]
    if ctx.alert:
        lines += ["", f"ALERT [{ctx.alert['error_class']}] at {ctx.error_stage}", f"  {ctx.alert['message']}",
                  f"  Next step: {ctx.alert['suggested_action']}"]
        if ctx.alert.get("explanation"):
            lines.append(f"  Claude says: {ctx.alert['explanation']}")
    s = ctx.stats
    if "rows_extracted" in s:
        lines += ["", f"Rows read: {s.get('rows_extracted')}   written: {s.get('rows_written')}   rejected: {s.get('rows_rejected', 0)}"]
    return "\n".join(lines)
