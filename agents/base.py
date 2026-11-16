"""
Shared building blocks for the agents.

Every agent follows the same contract:
    agent.run(ctx) -> AgentResult

`ctx` (RunContext) is the shared "memory" of one pipeline run. Each agent reads
what earlier agents produced and adds its own output, so the orchestrator
(pipeline.py) only has to call them in order.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Where alerts, run metadata and rejected rows are written.
LOG_DIR = Path(os.environ.get("ETL_LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"


@dataclass
class Finding:
    """One thing an agent noticed. ERROR findings stop the run."""

    severity: str
    code: str
    message: str
    column: Optional[str] = None

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        return f"{self.severity}: {self.code}{where} - {self.message}"


@dataclass
class AgentResult:
    agent: str
    ok: bool
    summary: str = ""
    findings: List[Finding] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == WARNING]


@dataclass
class ColumnInfo:
    name: str
    type_str: str
    category: str  # integer | decimal | string | datetime | date | time | boolean | binary | other
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    nullable: bool = True
    has_default: bool = False
    autoincrement: bool = False


@dataclass
class RunContext:
    """Shared state for one pipeline run."""

    job_name: Optional[str] = None
    request_text: Optional[str] = None
    full_reload: bool = False
    dry_run: bool = False

    job_cfg: Dict[str, Any] = field(default_factory=dict)
    source_engine: Any = None
    dest_engine: Any = None

    # filled in by the agents
    source_schema: Dict[str, ColumnInfo] = field(default_factory=dict)
    dest_schema: Dict[str, ColumnInfo] = field(default_factory=dict)
    dest_primary_key: List[str] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=dict)  # source column -> destination column
    unmapped_source: List[str] = field(default_factory=list)
    unmapped_dest: List[str] = field(default_factory=list)
    transform_steps: List[str] = field(default_factory=list)

    stats: Dict[str, Any] = field(default_factory=dict)
    results: List[AgentResult] = field(default_factory=list)
    alert: Optional[Dict[str, Any]] = None

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None
    status: str = "pending"  # pending | success | failed | dry_run_ok
    error: Optional[BaseException] = None
    error_stage: Optional[str] = None

    @property
    def all_findings(self) -> List[Finding]:
        out: List[Finding] = []
        for r in self.results:
            out.extend(r.findings)
        return out


class Agent:
    """Base class: subclasses implement `_run`, timing is handled here."""

    name = "agent"

    def run(self, ctx: RunContext) -> AgentResult:
        t0 = time.perf_counter()
        result = self._run(ctx)
        result.agent = self.name
        result.duration_s = round(time.perf_counter() - t0, 3)
        return result

    def _run(self, ctx: RunContext) -> AgentResult:  # pragma: no cover - interface
        raise NotImplementedError


class DataQualityError(Exception):
    """Raised when extracted rows cannot be loaded safely (bad types, NULLs, too long)."""


class StageFailed(Exception):
    """Raised by the orchestrator when an agent reports ok=False."""

    def __init__(self, stage: str, result: AgentResult):
        self.stage = stage
        self.result = result
        details = "; ".join(str(f) for f in result.errors) or result.summary
        super().__init__(f"{stage} stopped the run: {details}")
