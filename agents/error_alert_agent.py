"""
Error and Alert Agent.

When any stage fails it:
  1. classifies the failure (connection, missing object, schema change, data type,
     truncation, duplicate key, NULL violation, data quality, validation, ...)
  2. builds an alert with a plain "what to do next"
  3. optionally asks Claude for a short plain-English explanation (if an API key is set)
  4. delivers it: always to logs/alerts.log, and to a webhook (Slack/Teams style)
     and/or email if the matching environment variables are set
  5. never raises, and never includes passwords in what it sends

Environment variables (all optional):
  ETL_ALERT_WEBHOOK_URL                      POST {"text": "..."} to this URL
  ETL_ALERT_EMAIL_TO, ETL_ALERT_EMAIL_FROM,
  ETL_SMTP_HOST, ETL_SMTP_PORT (587), ETL_SMTP_USER, ETL_SMTP_PASSWORD
"""

from __future__ import annotations

import json
import logging
import os
import re
import smtplib
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any, Dict, Optional

from .base import LOG_DIR, DataQualityError, RunContext, StageFailed
from .llm import ask_claude

log = logging.getLogger(__name__)

# (error class, regex on the lower-cased message, what to do)
_PATTERNS = [
    ("CONNECTION", r"login failed|communication link|cannot open server|timeout expired|connection refused|tcp provider|could not connect",
     "Check that the SQL Server container/service is running, the server name and port, and the login in config.py or the environment variables."),
    ("MISSING_OBJECT", r"invalid object name|no such table|does not exist",
     "A table in the job does not exist in that database. Check the table names and run the destination CREATE TABLE script."),
    ("SCHEMA_DRIFT", r"invalid column name|no such column|has no column named",
     "A column the pipeline expects is missing. The source or destination table changed; compare the two schemas."),
    ("TRUNCATION", r"string or binary data would be truncated|data too long",
     "A text value is longer than the destination column. Widen the destination column or shorten the data."),
    ("DUPLICATE_KEY", r"violation of primary key|duplicate key|unique constraint|cannot insert duplicate",
     "Two rows share the same key. Check the primary_key in the job and the source data for duplicates."),
    ("NULL_VIOLATION", r"cannot insert the value null|not null constraint|null into column",
     "A required (NOT NULL) column received an empty value. Fix the source data or allow NULL in the destination."),
    ("TYPE_CONVERSION", r"conversion failed|error converting data type|invalid .* conversion|cannot convert",
     "A value could not be converted to the destination type. Check the data types in both tables."),
]
_ACTIONS = {
    "VALIDATION_FAILED": "Fix the errors listed above (they were found before any data moved), then run the job again. Nothing was loaded.",
    "BAD_REQUIREMENT": "Fix the job definition (config.JOBS) or the request text, then run again.",
    "DATA_QUALITY": "Open the rejected-rows CSV named in the message, fix those rows in the source, then run again. The watermark did not move.",
    "UNKNOWN": "Read the error message below; the watermark did not move, so it is safe to run the job again once fixed.",
}


def redact(text: str) -> str:
    """Remove passwords and API keys from anything that is logged or sent."""
    text = re.sub(r"(?i)(pwd|password)\s*=\s*[^;\s]+", r"\1=***", text)
    return re.sub(r"sk-ant-[A-Za-z0-9_\-]+", "sk-ant-***", text)


def classify(exc: BaseException) -> str:
    if isinstance(exc, DataQualityError):
        return "DATA_QUALITY"
    if isinstance(exc, StageFailed):
        return "BAD_REQUIREMENT" if exc.stage.startswith("Requirement") else "VALIDATION_FAILED"
    message = str(exc).lower()
    for name, pattern, _ in _PATTERNS:
        if re.search(pattern, message):
            return name
    return "UNKNOWN"


def suggested_action(error_class: str) -> str:
    for name, _, action in _PATTERNS:
        if name == error_class:
            return action
    return _ACTIONS.get(error_class, _ACTIONS["UNKNOWN"])


class ErrorAlertAgent:
    name = "Error and Alert Agent"

    def __init__(self, llm=None):
        self._llm = llm or ask_claude

    def handle(self, ctx: RunContext, exc: BaseException) -> Dict[str, Any]:
        error_class = classify(exc)
        message = redact(str(exc))
        alert: Dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": ctx.run_id,
            "job": ctx.job_name,
            "stage": ctx.error_stage,
            "error_class": error_class,
            "message": message[:1500],
            "suggested_action": suggested_action(error_class),
            "delivered_to": [],
        }
        explanation = self._explain(alert)
        if explanation:
            alert["explanation"] = redact(explanation)

        self._to_file(alert)
        text = self._format(alert)
        if self._to_webhook(text):
            alert["delivered_to"].append("webhook")
        if self._to_email(text, alert):
            alert["delivered_to"].append("email")
        log.error("ALERT %s in %s: %s", error_class, ctx.error_stage, message[:300])
        return alert

    # -------------------------------------------------------------- helpers
    def _explain(self, alert: Dict[str, Any]) -> Optional[str]:
        try:
            return self._llm(
                "You explain data-pipeline failures to a non-expert in 2 short sentences, then give one concrete next step. No jargon, no markdown.",
                f"Job: {alert['job']}\nStage: {alert['stage']}\nClass: {alert['error_class']}\nError: {alert['message']}",
                max_tokens=250,
            )
        except Exception:
            return None

    @staticmethod
    def _format(alert: Dict[str, Any]) -> str:
        lines = [
            f"ETL ALERT [{alert['error_class']}] job={alert['job']} stage={alert['stage']} run={alert['run_id']}",
            f"What happened: {alert['message']}",
            f"What to do: {alert['suggested_action']}",
        ]
        if alert.get("explanation"):
            lines.append(f"Plain English: {alert['explanation']}")
        return "\n".join(lines)

    @staticmethod
    def _to_file(alert: Dict[str, Any]) -> None:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(LOG_DIR / "alerts.log", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert) + "\n")
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("could not write alerts.log: %s", exc)

    @staticmethod
    def _to_webhook(text: str) -> bool:
        url = os.environ.get("ETL_ALERT_WEBHOOK_URL")
        if not url:
            return False
        try:
            req = urllib.request.Request(
                url, data=json.dumps({"text": text}).encode("utf-8"), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            return True
        except Exception as exc:
            log.warning("webhook alert failed: %s", exc)
            return False

    @staticmethod
    def _to_email(text: str, alert: Dict[str, Any]) -> bool:
        to, host = os.environ.get("ETL_ALERT_EMAIL_TO"), os.environ.get("ETL_SMTP_HOST")
        if not to or not host:
            return False
        try:
            msg = EmailMessage()
            msg["Subject"] = f"[ETL ALERT] {alert['job']}: {alert['error_class']}"
            msg["From"] = os.environ.get("ETL_ALERT_EMAIL_FROM", os.environ.get("ETL_SMTP_USER", "etl-alerts@localhost"))
            msg["To"] = to
            msg.set_content(text)
            with smtplib.SMTP(host, int(os.environ.get("ETL_SMTP_PORT", "587")), timeout=15) as smtp:
                smtp.starttls()
                user = os.environ.get("ETL_SMTP_USER")
                if user:
                    smtp.login(user, os.environ.get("ETL_SMTP_PASSWORD", ""))
                smtp.send_message(msg)
            return True
        except Exception as exc:
            log.warning("email alert failed: %s", exc)
            return False
