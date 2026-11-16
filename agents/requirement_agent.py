"""
Requirement Agent: turns "what the user wants" into a checked job specification.

Two ways in:
  * a job name from config.JOBS  (always available, no AI needed)
  * a plain-English request      (needs ANTHROPIC_API_KEY; Claude PROPOSES the
                                   spec, this agent and the Validation Agent
                                   VERIFY it before anything runs)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .base import ERROR, INFO, Agent, AgentResult, Finding, RunContext
from .llm import ask_claude

VALID_WRITE = {"full": {"truncate_load", "upsert"}, "incremental": {"append", "upsert"}}
VALID_WATERMARK = {"datetime", "integer", "rowversion"}
ALLOWED_KEYS = {
    "source_table", "dest_table", "load_type", "write_strategy", "primary_key",
    "watermark_column", "watermark_type", "column_map", "exclude_columns",
    "allow_casts", "trim_strings", "on_bad_rows",
}

SYSTEM_PROMPT = (
    "You convert a data-loading request into a JSON job specification. "
    "Reply with ONE JSON object and nothing else. Allowed keys: "
    "source_table, dest_table, load_type (full|incremental), "
    "write_strategy (full: truncate_load|upsert; incremental: append|upsert), "
    "primary_key (list of destination column names, required for upsert), "
    "watermark_column and watermark_type (datetime|integer|rowversion; required for incremental). "
    "Use only table names from the provided list. If the request is unclear, "
    "make the safest choice (upsert with a primary key) and never invent table names."
)


def validate_job_cfg(cfg: Dict[str, Any]) -> List[Finding]:
    """Check a job specification for completeness and internal consistency."""
    f: List[Finding] = []
    for key in ("source_table", "dest_table", "load_type", "write_strategy"):
        if not cfg.get(key):
            f.append(Finding(ERROR, "MISSING_FIELD", f"'{key}' is required"))
    load_type = cfg.get("load_type")
    if load_type and load_type not in VALID_WRITE:
        f.append(Finding(ERROR, "BAD_LOAD_TYPE", f"load_type must be 'full' or 'incremental', got '{load_type}'"))
        return f
    strategy = cfg.get("write_strategy")
    if load_type and strategy and strategy not in VALID_WRITE[load_type]:
        allowed = " or ".join(sorted(VALID_WRITE[load_type]))
        f.append(Finding(ERROR, "BAD_WRITE_STRATEGY", f"a {load_type} load must use {allowed}, got '{strategy}'"))
    if strategy == "upsert" and not cfg.get("primary_key"):
        f.append(Finding(ERROR, "MISSING_PRIMARY_KEY", "upsert needs 'primary_key' (list of destination columns)"))
    if load_type == "incremental":
        if not cfg.get("watermark_column"):
            f.append(Finding(ERROR, "MISSING_WATERMARK", "an incremental load needs 'watermark_column'"))
        if cfg.get("watermark_type") not in VALID_WATERMARK:
            f.append(Finding(ERROR, "BAD_WATERMARK_TYPE", "watermark_type must be datetime, integer or rowversion"))
    if cfg.get("on_bad_rows", "fail") not in ("fail", "quarantine"):
        f.append(Finding(ERROR, "BAD_ON_BAD_ROWS", "on_bad_rows must be 'fail' or 'quarantine'"))
    return f


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class RequirementAgent(Agent):
    name = "Requirement Agent"

    def __init__(self, jobs: Optional[Dict[str, Dict[str, Any]]] = None, llm=None):
        self._jobs = jobs
        self._llm = llm or ask_claude

    def _jobs_config(self) -> Dict[str, Dict[str, Any]]:
        if self._jobs is not None:
            return self._jobs
        from config import JOBS  # lazy: keeps this module importable without a config

        return JOBS

    def _run(self, ctx: RunContext) -> AgentResult:
        findings: List[Finding] = []
        if ctx.request_text:
            cfg = self._from_text(ctx, findings)
            if cfg is None:
                return AgentResult(self.name, False, "could not turn the request into a job", findings)
            slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{cfg.get('source_table', '')}_to_{cfg.get('dest_table', '')}").strip("_")
            ctx.job_name = ctx.job_name or f"adhoc_{slug}"[:150]
            findings.append(Finding(INFO, "FROM_TEXT", "job specification proposed by Claude from the request; the next agents will verify it"))
        else:
            jobs = self._jobs_config()
            if not ctx.job_name or ctx.job_name not in jobs:
                known = ", ".join(sorted(jobs)) or "none"
                findings.append(Finding(ERROR, "UNKNOWN_JOB", f"job '{ctx.job_name}' is not defined. Known jobs: {known}"))
                return AgentResult(self.name, False, "unknown job", findings)
            cfg = dict(jobs[ctx.job_name])

        findings.extend(validate_job_cfg(cfg))
        ok = not any(x.severity == ERROR for x in findings)
        if ok:
            ctx.job_cfg = cfg
        summary = (
            f"{ctx.job_name}: {cfg.get('load_type')} load {cfg.get('source_table')} -> "
            f"{cfg.get('dest_table')} ({cfg.get('write_strategy')})"
        )
        return AgentResult(self.name, ok, summary, findings)

    def _from_text(self, ctx: RunContext, findings: List[Finding]) -> Optional[Dict[str, Any]]:
        tables = ""
        try:
            from sqlalchemy import inspect

            names = inspect(ctx.source_engine).get_table_names()
            tables = "Available source tables: " + ", ".join(names) + "\n"
        except Exception:
            pass
        answer = self._llm(SYSTEM_PROMPT, f"{tables}Request: {ctx.request_text}")
        if answer is None:
            findings.append(Finding(ERROR, "NO_LLM", "plain-English requests need ANTHROPIC_API_KEY and the 'anthropic' package; use --job instead"))
            return None
        data = _extract_json(answer)
        if data is None:
            findings.append(Finding(ERROR, "BAD_LLM_REPLY", "Claude did not return a JSON job specification"))
            return None
        return {k: v for k, v in data.items() if k in ALLOWED_KEYS}
