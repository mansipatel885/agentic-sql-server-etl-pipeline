"""
Source and Destination Validation Agent.

Looks at the real source and destination tables (not just the config) and stops
the run, with a clear message, BEFORE any data moves if something would break:
missing tables, a new source column, incompatible data types, values that will
not fit (length / precision), missing keys, or a bad watermark column.
"""

from __future__ import annotations

import difflib
from typing import Dict, List, Optional

from .base import ERROR, INFO, WARNING, Agent, AgentResult, ColumnInfo, Finding, RunContext
from .schema_utils import build_column_pairs, reflect_table

_INT_RANK = (("BIGINT", 4), ("SMALLINT", 2), ("TINYINT", 1), ("INT", 3))
_WATERMARK_CATEGORY = {"datetime": "datetime", "integer": "integer", "rowversion": "binary"}


def _int_rank(info: ColumnInfo) -> int:
    upper = info.type_str.upper()
    for token, rank in _INT_RANK:
        if token in upper:
            return rank
    return 3


def _suggest_ddl(table: str, schema: Dict[str, ColumnInfo]) -> str:
    cols = ", ".join(f"{c.name} {c.type_str} {'NULL' if c.nullable else 'NOT NULL'}" for c in schema.values())
    return f"CREATE TABLE {table} ({cols})"


def compare_columns(src: ColumnInfo, dst: ColumnInfo, allow_cast: bool) -> List[Finding]:
    """Compare one source column with its destination column."""
    label = f"{src.name} ({src.type_str}) -> {dst.name} ({dst.type_str})"
    sc, dc = src.category, dst.category
    out: List[Finding] = []

    if sc == dc:
        if sc == "string":
            if src.length is None and dst.length is not None:
                out.append(Finding(WARNING, "LENGTH_UNBOUNDED_SOURCE", f"{label}: source text has no length limit, destination allows {dst.length}; rows are checked for length at load time", dst.name))
            elif src.length is not None and dst.length is not None and src.length > dst.length:
                out.append(Finding(ERROR, "LENGTH_OVERFLOW", f"{label}: source allows {src.length} characters, destination only {dst.length}. Widen the destination column.", dst.name))
        elif sc == "decimal":
            if None not in (src.precision, dst.precision) and src.precision > dst.precision:
                out.append(Finding(ERROR, "PRECISION_OVERFLOW", f"{label}: source precision {src.precision} exceeds destination precision {dst.precision}", dst.name))
            elif None not in (src.scale, dst.scale) and src.scale > dst.scale:
                out.append(Finding(ERROR, "SCALE_OVERFLOW", f"{label}: source keeps {src.scale} decimal places, destination only {dst.scale}", dst.name))
        elif sc == "integer" and _int_rank(src) > _int_rank(dst):
            out.append(Finding(WARNING, "INTEGER_RANGE", f"{label}: destination integer type is smaller than the source", dst.name))
        return out

    safe = {("integer", "decimal"), ("date", "datetime")}
    lossy = {("decimal", "integer"), ("datetime", "date"), ("datetime", "time")}
    if (sc, dc) in safe:
        out.append(Finding(INFO, "WIDENING_CAST", f"{label}: converted from {sc} to {dc}", dst.name))
    elif (sc, dc) in lossy:
        out.append(Finding(WARNING, "LOSSY_CAST", f"{label}: converting {sc} to {dc} can lose information", dst.name))
    elif dc == "string":
        out.append(Finding(WARNING, "STORED_AS_TEXT", f"{label}: {sc} values will be stored as text", dst.name))
    elif sc == "string" and dc in {"integer", "decimal", "datetime", "date"}:
        sev = WARNING if allow_cast else ERROR
        hint = "" if allow_cast else " Add the destination column to 'allow_casts' if the text always parses."
        out.append(Finding(sev, "TYPE_MISMATCH", f"{label}: text must be parsed as {dc}.{hint}", dst.name))
    else:
        out.append(Finding(ERROR, "TYPE_MISMATCH", f"{label}: {sc} cannot be loaded into {dc}", dst.name))
    return out


class ValidationAgent(Agent):
    name = "Validation Agent"

    def _run(self, ctx: RunContext) -> AgentResult:
        cfg = ctx.job_cfg
        findings: List[Finding] = []

        src = reflect_table(ctx.source_engine, cfg["source_table"])
        dst = reflect_table(ctx.dest_engine, cfg["dest_table"])
        if src is None:
            findings.append(Finding(ERROR, "SOURCE_TABLE_MISSING", f"source table {cfg['source_table']} does not exist"))
        if dst is None:
            hint = f" Suggested: {_suggest_ddl(cfg['dest_table'], src[0])}" if src is not None else ""
            findings.append(Finding(ERROR, "DEST_TABLE_MISSING", f"destination table {cfg['dest_table']} does not exist.{hint}"))
        if src is None or dst is None:
            return AgentResult(self.name, False, "table check failed", findings)

        ctx.source_schema, _ = src
        ctx.dest_schema, ctx.dest_primary_key = dst

        mapping, unmapped_src, unmapped_dst = build_column_pairs(
            list(ctx.source_schema), list(ctx.dest_schema), cfg.get("column_map"), cfg.get("exclude_columns")
        )

        findings += self._check_keys(ctx, mapping)
        findings += self._check_watermark(ctx)
        findings += self._check_columns(ctx, mapping, unmapped_src, unmapped_dst)

        ok = not any(f.severity == ERROR for f in findings)
        n_err = sum(f.severity == ERROR for f in findings)
        n_warn = sum(f.severity == WARNING for f in findings)
        summary = (
            f"{len(mapping)} columns compatible, {n_err} error(s), {n_warn} warning(s)"
            if ok or n_err else f"{len(mapping)} columns checked"
        )
        return AgentResult(self.name, ok, summary, findings)

    # ------------------------------------------------------------------ keys
    def _check_keys(self, ctx: RunContext, mapping: Dict[str, str]) -> List[Finding]:
        cfg, out = ctx.job_cfg, []
        if cfg.get("write_strategy") != "upsert":
            return out
        dest_cols = set(ctx.dest_schema)
        for key in cfg.get("primary_key", []):
            if key not in dest_cols:
                out.append(Finding(ERROR, "PRIMARY_KEY_MISSING_IN_DEST", f"primary key column '{key}' is not in the destination table", key))
            elif key not in mapping.values():
                out.append(Finding(ERROR, "PRIMARY_KEY_NOT_LOADED", f"primary key column '{key}' has no matching source column", key))
        if not ctx.dest_primary_key:
            out.append(Finding(WARNING, "DEST_HAS_NO_PRIMARY_KEY", "destination has no primary key constraint; duplicates are possible if the MERGE key is not unique"))
        return out

    # ------------------------------------------------------------- watermark
    def _check_watermark(self, ctx: RunContext) -> List[Finding]:
        cfg, out = ctx.job_cfg, []
        if cfg.get("load_type") != "incremental":
            return out
        col = cfg["watermark_column"]
        info: Optional[ColumnInfo] = ctx.source_schema.get(col)
        if info is None:
            out.append(Finding(ERROR, "WATERMARK_MISSING", f"watermark column '{col}' is not in the source table", col))
            return out
        expected = _WATERMARK_CATEGORY[cfg["watermark_type"]]
        if info.category != expected:
            out.append(Finding(ERROR, "WATERMARK_TYPE_MISMATCH", f"watermark '{col}' is {info.category} ({info.type_str}) but the job says {cfg['watermark_type']}", col))
        if info.nullable:
            out.append(Finding(WARNING, "WATERMARK_NULLABLE", f"watermark '{col}' allows NULL; rows with NULL are never picked up by an incremental load", col))
        return out

    # --------------------------------------------------------------- columns
    def _check_columns(self, ctx, mapping, unmapped_src, unmapped_dst) -> List[Finding]:
        cfg, out = ctx.job_cfg, []
        allow = {c.lower() for c in cfg.get("allow_casts", [])}

        for s in unmapped_src:
            close = difflib.get_close_matches(s, unmapped_dst, n=1, cutoff=0.6)
            hint = f" Did you mean '{close[0]}'? Map it with column_map." if close else ""
            alter = f"ALTER TABLE {cfg['dest_table']} ADD {s} {ctx.source_schema[s].type_str} NULL"
            out.append(Finding(
                ERROR, "NEW_COLUMN_IN_SOURCE",
                f"source column '{s}' has no destination column.{hint} To keep it: {alter}. To ignore it: add '{s}' to exclude_columns.",
                s,
            ))
        for d in unmapped_dst:
            info = ctx.dest_schema[d]
            if not info.nullable and not info.has_default and not info.autoincrement:
                out.append(Finding(ERROR, "DEST_REQUIRED_COLUMN_NOT_LOADED", f"destination column '{d}' is NOT NULL with no default, but nothing is loaded into it", d))
            else:
                out.append(Finding(INFO, "DEST_COLUMN_NOT_IN_SOURCE", f"destination column '{d}' has no source column and will stay empty", d))
        for s, d in mapping.items():
            out.extend(compare_columns(ctx.source_schema[s], ctx.dest_schema[d], d.lower() in allow))
        return out
