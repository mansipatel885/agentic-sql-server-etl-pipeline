"""
Schema helpers shared by the Validation, Mapping and Transformation agents.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

from sqlalchemy import inspect
from sqlalchemy import types as T

from .base import ColumnInfo


def split_table(name: str) -> Tuple[Optional[str], str]:
    """'dbo.Orders' -> ('dbo', 'Orders'); 'Orders' -> (None, 'Orders')."""
    if "." in name:
        schema, table = name.split(".", 1)
        return schema, table
    return None, name


def type_category(sa_type) -> str:
    """Collapse a database type into a small set of comparable categories."""
    type_name = type(sa_type).__name__.upper()
    if type_name == "BIT" or isinstance(sa_type, T.Boolean):
        return "boolean"
    if isinstance(sa_type, T.Integer):
        return "integer"
    if isinstance(sa_type, T.Numeric):  # includes Float, DECIMAL, MONEY-like
        return "decimal"
    if isinstance(sa_type, T.DateTime):
        return "datetime"
    if isinstance(sa_type, T.Date):
        return "date"
    if isinstance(sa_type, T.Time):
        return "time"
    if isinstance(sa_type, (T.String, T.Text)):
        return "string"
    if isinstance(sa_type, T.LargeBinary) or type_name in {"BINARY", "VARBINARY", "ROWVERSION", "TIMESTAMP"}:
        return "binary"
    return "other"


def reflect_table(engine, table_name: str) -> Optional[Tuple[Dict[str, ColumnInfo], List[str]]]:
    """
    Read a table's columns and primary key. Returns None if the table does not exist.
    Columns come back in table order.
    """
    schema, table = split_table(table_name)
    inspector = inspect(engine)
    if not inspector.has_table(table, schema=schema):
        return None

    columns: Dict[str, ColumnInfo] = {}
    for col in inspector.get_columns(table, schema=schema):
        sa_type = col["type"]
        try:
            type_str = sa_type.compile(dialect=engine.dialect)
        except Exception:
            type_str = str(sa_type)
        length = getattr(sa_type, "length", None)
        columns[col["name"]] = ColumnInfo(
            name=col["name"],
            type_str=type_str,
            category=type_category(sa_type),
            length=length if isinstance(length, int) and length > 0 else None,
            precision=getattr(sa_type, "precision", None),
            scale=getattr(sa_type, "scale", None),
            nullable=bool(col.get("nullable", True)),
            has_default=col.get("default") is not None,
            autoincrement=bool(col.get("autoincrement")) and col.get("autoincrement") != "auto",
        )
    pk = inspector.get_pk_constraint(table, schema=schema).get("constrained_columns", []) or []
    return columns, list(pk)


def build_column_pairs(
    source_cols: List[str],
    dest_cols: List[str],
    column_map: Optional[Dict[str, str]] = None,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Pair source columns with destination columns.

    Rules, in order:
      1. columns listed in `exclude_columns` are skipped on purpose
      2. an explicit `column_map` entry (source name -> destination name) wins
      3. otherwise names are matched case-insensitively

    Returns (mapping, unmapped_source, unmapped_dest).
    """
    column_map = column_map or {}
    exclude = {c.lower() for c in (exclude_columns or [])}
    dest_lookup = {c.lower(): c for c in dest_cols}

    mapping: Dict[str, str] = {}
    unmapped_source: List[str] = []
    for src in source_cols:
        if src.lower() in exclude:
            continue
        explicit = column_map.get(src)
        if explicit is not None:
            target = dest_lookup.get(explicit.lower())
        else:
            target = dest_lookup.get(src.lower())
        if target is None:
            unmapped_source.append(src)
        else:
            mapping[src] = target

    used = set(mapping.values())
    unmapped_dest = [c for c in dest_cols if c not in used]
    return mapping, unmapped_source, unmapped_dest


def schema_fingerprint(schema: Dict[str, ColumnInfo]) -> str:
    """A short hash of column names and types, used to notice schema changes between runs."""
    text = "|".join(f"{c.name}:{c.type_str}:{int(c.nullable)}" for c in schema.values())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
