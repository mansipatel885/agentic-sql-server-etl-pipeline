"""
Data Mapping Agent: fixes the final source-column -> destination-column map
and records which pairs need a type conversion.

The mapping rules (explicit `column_map` first, then case-insensitive name
match, `exclude_columns` skipped on purpose) live in schema_utils so the
Validation Agent and this agent always agree.
"""

from __future__ import annotations

from typing import List

from .base import INFO, Agent, AgentResult, Finding, RunContext
from .schema_utils import build_column_pairs


class MappingAgent(Agent):
    name = "Data Mapping Agent"

    def _run(self, ctx: RunContext) -> AgentResult:
        cfg = ctx.job_cfg
        mapping, unmapped_src, unmapped_dst = build_column_pairs(
            list(ctx.source_schema), list(ctx.dest_schema), cfg.get("column_map"), cfg.get("exclude_columns")
        )
        ctx.mapping, ctx.unmapped_source, ctx.unmapped_dest = mapping, unmapped_src, unmapped_dst

        renamed = {s: d for s, d in mapping.items() if s != d}
        conversions = {
            s: (ctx.source_schema[s].category, ctx.dest_schema[d].category)
            for s, d in mapping.items()
            if ctx.source_schema[s].category != ctx.dest_schema[d].category
        }
        ctx.stats["renamed_columns"] = renamed
        ctx.stats["conversions"] = conversions

        findings: List[Finding] = []
        for s, d in renamed.items():
            findings.append(Finding(INFO, "RENAMED", f"{s} -> {d}", d))
        for s, (a, b) in conversions.items():
            findings.append(Finding(INFO, "CONVERSION", f"{s}: {a} -> {b}", mapping[s]))

        summary = f"{len(mapping)} columns mapped, {len(renamed)} renamed, {len(conversions)} converted"
        return AgentResult(self.name, True, summary, findings)
