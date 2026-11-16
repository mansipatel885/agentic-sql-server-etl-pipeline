"""
Tests for the agentic pipeline. They use two in-memory SQLite databases as
stand-ins for the source and destination, so they run anywhere with no SQL
Server. (The SQL Server specific write path, MERGE + control table, is the
code that already ran against Docker; see AGENTIC_README.md for the on-your-Mac check.)

Run:   python -m unittest discover -s tests -v
"""

import http.server
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

LOG_TMP = tempfile.mkdtemp(prefix="etl_agent_tests_")
os.environ["ETL_LOG_DIR"] = LOG_TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("pyodbc", types.ModuleType("pyodbc"))  # not needed here; db.py imports it

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from agents import (  # noqa: E402
    DataQualityError, ErrorAlertAgent, ExecutionAgent, MappingAgent, MetadataAgent,
    RequirementAgent, RunContext, TransformationAgent, ValidationAgent,
)
from agents import error_alert_agent  # noqa: E402
from agents.error_alert_agent import classify, redact  # noqa: E402
from pipeline import AgenticPipeline  # noqa: E402


def engines(src_sql=(), dst_sql=()):
    src, dst = create_engine("sqlite://"), create_engine("sqlite://")
    for eng, stmts in ((src, src_sql), (dst, dst_sql)):
        with eng.begin() as c:
            for s in stmts:
                c.execute(text(s))
    return src, dst


def codes(result, severity=None):
    return {f.code for f in result.findings if severity is None or f.severity == severity}


def prepared_ctx(cfg, src, dst):
    """Run requirement/validation/mapping/transformation-plan and return the context."""
    ctx = RunContext(job_name="t", job_cfg=cfg, source_engine=src, dest_engine=dst)
    ValidationAgent().run(ctx)
    MappingAgent().run(ctx)
    TransformationAgent().run(ctx)
    return ctx


class FakeControl:
    """In-memory stand-in for control.py (which talks to SQL Server through pyodbc)."""

    def __init__(self):
        self.watermark, self.status, self.calls = None, None, []

    def ensure_job_registered(self, job, wm_type): self.calls.append("ensure")
    def reset_watermark(self, job): self.watermark = None; self.calls.append("reset")
    def mark_run_start(self, job): self.status = "running"; return datetime.now(timezone.utc)
    def get_last_watermark(self, job): return self.watermark
    def mark_run_success(self, job, wm): self.status = "success"; self.watermark = None if wm is None else str(wm)
    def mark_run_failed(self, job): self.status = "failed"


# --------------------------------------------------------------------------- 1
class RequirementTests(unittest.TestCase):
    JOBS = {
        "ok_full": {"source_table": "a", "dest_table": "b", "load_type": "full", "write_strategy": "upsert", "primary_key": ["id"]},
        "no_pk": {"source_table": "a", "dest_table": "b", "load_type": "full", "write_strategy": "upsert"},
        "no_wm": {"source_table": "a", "dest_table": "b", "load_type": "incremental", "write_strategy": "append"},
        "bad_strategy": {"source_table": "a", "dest_table": "b", "load_type": "incremental", "write_strategy": "truncate_load", "watermark_column": "x", "watermark_type": "integer"},
    }

    def run_job(self, name, **kw):
        return RequirementAgent(jobs=self.JOBS, **kw).run(RunContext(job_name=name))

    def test_valid_job(self):
        self.assertTrue(self.run_job("ok_full").ok)

    def test_unknown_job(self):
        self.assertIn("UNKNOWN_JOB", codes(self.run_job("nope")))

    def test_upsert_needs_key(self):
        self.assertIn("MISSING_PRIMARY_KEY", codes(self.run_job("no_pk")))

    def test_incremental_needs_watermark(self):
        r = self.run_job("no_wm")
        self.assertFalse(r.ok)
        self.assertTrue({"MISSING_WATERMARK", "BAD_WATERMARK_TYPE"} <= codes(r))

    def test_wrong_strategy_for_load_type(self):
        self.assertIn("BAD_WRITE_STRATEGY", codes(self.run_job("bad_strategy")))

    def test_text_request_with_llm_reply(self):
        reply = 'Sure! {"source_table":"Orders","dest_table":"Orders","load_type":"incremental","write_strategy":"upsert","primary_key":["OrderID"],"watermark_column":"ModifiedDate","watermark_type":"datetime","evil":"drop table"}'
        ctx = RunContext(request_text="load new orders", source_engine=create_engine("sqlite://"))
        r = RequirementAgent(jobs={}, llm=lambda s, p, **k: reply).run(ctx)
        self.assertTrue(r.ok, r.findings)
        self.assertNotIn("evil", ctx.job_cfg)  # unknown keys from the LLM are dropped
        self.assertTrue(ctx.job_name.startswith("adhoc_"))

    def test_text_request_without_llm(self):
        r = RequirementAgent(jobs={}, llm=lambda s, p, **k: None).run(RunContext(request_text="x", source_engine=create_engine("sqlite://")))
        self.assertIn("NO_LLM", codes(r))

    def test_text_request_with_garbage_reply(self):
        r = RequirementAgent(jobs={}, llm=lambda s, p, **k: "I cannot do that").run(RunContext(request_text="x", source_engine=create_engine("sqlite://")))
        self.assertIn("BAD_LLM_REPLY", codes(r))

    def test_llm_proposal_is_still_validated(self):
        reply = '{"source_table":"a","dest_table":"b","load_type":"incremental","write_strategy":"upsert"}'
        r = RequirementAgent(jobs={}, llm=lambda s, p, **k: reply).run(RunContext(request_text="x", source_engine=create_engine("sqlite://")))
        self.assertFalse(r.ok)


# --------------------------------------------------------------------------- 2
CFG = {"source_table": "Orders", "dest_table": "Orders", "load_type": "full", "write_strategy": "upsert", "primary_key": ["OrderID"]}
SRC_ORDERS = "CREATE TABLE Orders (OrderID INTEGER PRIMARY KEY, CustomerID INTEGER, Note VARCHAR(20), Amount NUMERIC(10,2), ModifiedDate DATETIME NOT NULL)"


class ValidationTests(unittest.TestCase):
    def validate(self, src_sql, dst_sql, **cfg_over):
        src, dst = engines([src_sql], [dst_sql] if dst_sql else [])
        ctx = RunContext(job_name="t", job_cfg={**CFG, **cfg_over}, source_engine=src, dest_engine=dst)
        return ValidationAgent().run(ctx), ctx

    def test_matching_schemas_pass(self):
        r, _ = self.validate(SRC_ORDERS, SRC_ORDERS)
        self.assertTrue(r.ok, r.findings)

    def test_new_source_column_is_caught_with_fix(self):
        r, _ = self.validate(SRC_ORDERS.replace("Amount", "Discount NUMERIC(5,2), Amount"), SRC_ORDERS)
        self.assertFalse(r.ok)
        msg = next(f.message for f in r.findings if f.code == "NEW_COLUMN_IN_SOURCE")
        self.assertIn("ALTER TABLE Orders ADD Discount", msg)
        self.assertIn("exclude_columns", msg)

    def test_new_column_can_be_excluded_on_purpose(self):
        r, _ = self.validate(SRC_ORDERS.replace("Amount", "Discount NUMERIC(5,2), Amount"), SRC_ORDERS, exclude_columns=["Discount"])
        self.assertTrue(r.ok, r.findings)

    def test_close_name_suggests_mapping(self):
        r, _ = self.validate(SRC_ORDERS.replace("CustomerID", "CustID"), SRC_ORDERS)
        msg = next(f.message for f in r.findings if f.code == "NEW_COLUMN_IN_SOURCE")
        self.assertIn("Did you mean 'CustomerID'", msg)

    def test_column_map_resolves_rename(self):
        r, _ = self.validate(SRC_ORDERS.replace("CustomerID", "CustID"), SRC_ORDERS, column_map={"CustID": "CustomerID"})
        self.assertTrue(r.ok, r.findings)

    def test_length_overflow(self):
        r, _ = self.validate(SRC_ORDERS, SRC_ORDERS.replace("VARCHAR(20)", "VARCHAR(10)"))
        self.assertIn("LENGTH_OVERFLOW", codes(r, "ERROR"))

    def test_precision_overflow(self):
        r, _ = self.validate(SRC_ORDERS, SRC_ORDERS.replace("NUMERIC(10,2)", "NUMERIC(6,2)"))
        self.assertIn("PRECISION_OVERFLOW", codes(r, "ERROR"))

    def test_text_into_integer_is_error_unless_allowed(self):
        dst = SRC_ORDERS.replace("Note VARCHAR(20)", "Note INTEGER")
        r, _ = self.validate(SRC_ORDERS, dst)
        self.assertIn("TYPE_MISMATCH", codes(r, "ERROR"))
        r2, _ = self.validate(SRC_ORDERS, dst, allow_casts=["Note"])
        self.assertTrue(r2.ok, r2.findings)

    def test_number_into_text_is_warning(self):
        r, _ = self.validate(SRC_ORDERS, SRC_ORDERS.replace("CustomerID INTEGER", "CustomerID VARCHAR(20)"))
        self.assertTrue(r.ok)
        self.assertIn("STORED_AS_TEXT", codes(r, "WARNING"))

    def test_missing_destination_gives_ddl(self):
        r, _ = self.validate(SRC_ORDERS, None)
        msg = next(f.message for f in r.findings if f.code == "DEST_TABLE_MISSING")
        self.assertIn("CREATE TABLE Orders", msg)

    def test_missing_source_table(self):
        src, dst = engines([], [SRC_ORDERS])
        r = ValidationAgent().run(RunContext(job_name="t", job_cfg=dict(CFG), source_engine=src, dest_engine=dst))
        self.assertIn("SOURCE_TABLE_MISSING", codes(r))

    def test_required_dest_column_not_loaded(self):
        r, _ = self.validate(SRC_ORDERS, SRC_ORDERS.replace("Note VARCHAR(20)", "Note VARCHAR(20), Region VARCHAR(5) NOT NULL"))
        self.assertIn("DEST_REQUIRED_COLUMN_NOT_LOADED", codes(r, "ERROR"))

    def test_primary_key_missing_in_dest(self):
        r, _ = self.validate(SRC_ORDERS, SRC_ORDERS, primary_key=["Nope"])
        self.assertIn("PRIMARY_KEY_MISSING_IN_DEST", codes(r, "ERROR"))

    def test_watermark_checks(self):
        inc = dict(load_type="incremental", write_strategy="upsert", watermark_column="ModifiedDate", watermark_type="datetime")
        self.assertTrue(self.validate(SRC_ORDERS, SRC_ORDERS, **inc)[0].ok)
        bad_col = self.validate(SRC_ORDERS, SRC_ORDERS, **{**inc, "watermark_column": "Nope"})[0]
        self.assertIn("WATERMARK_MISSING", codes(bad_col))
        bad_type = self.validate(SRC_ORDERS, SRC_ORDERS, **{**inc, "watermark_type": "integer"})[0]
        self.assertIn("WATERMARK_TYPE_MISMATCH", codes(bad_type))


# --------------------------------------------------------------------------- 3
class MappingTests(unittest.TestCase):
    def test_rename_and_case_insensitive(self):
        src, dst = engines(["CREATE TABLE s (id INTEGER, CustID INTEGER, name VARCHAR(5))"],
                           ["CREATE TABLE d (ID INTEGER, CustomerID INTEGER, Name VARCHAR(5))"])
        cfg = {"source_table": "s", "dest_table": "d", "load_type": "full", "write_strategy": "truncate_load", "column_map": {"CustID": "CustomerID"}}
        ctx = prepared_ctx(cfg, src, dst)
        self.assertEqual(ctx.mapping, {"id": "ID", "CustID": "CustomerID", "name": "Name"})
        self.assertEqual(ctx.stats["renamed_columns"]["CustID"], "CustomerID")


# --------------------------------------------------------------------------- 4
class TransformationTests(unittest.TestCase):
    def ctx_for(self, src_cols, dst_cols, **cfg_over):
        src, dst = engines([f"CREATE TABLE s ({src_cols})"], [f"CREATE TABLE d ({dst_cols})"])
        cfg = {"source_table": "s", "dest_table": "d", "load_type": "full", "write_strategy": "truncate_load", **cfg_over}
        ctx = prepared_ctx(cfg, src, dst)
        ctx.run_id, ctx.job_name = "tr1", "tjob"
        return ctx

    def test_trim_and_cast(self):
        ctx = self.ctx_for("id VARCHAR(5), name VARCHAR(10), amount VARCHAR(10)", "id INTEGER, name VARCHAR(10), amount NUMERIC(10,2)", allow_casts=["id", "amount"])
        df = pd.DataFrame({"id": ["1", "2"], "name": ["  ann ", "bob"], "amount": ["1.5", "2"]})
        out, rejected = TransformationAgent().apply(ctx, df)
        self.assertEqual(rejected, 0)
        self.assertEqual(list(out["name"]), ["ann", "bob"])
        self.assertEqual(list(out["id"]), [1, 2])
        self.assertEqual(list(out["amount"]), [1.5, 2.0])

    def test_bad_rows_fail_by_default_and_are_saved(self):
        ctx = self.ctx_for("id VARCHAR(5), name VARCHAR(10)", "id INTEGER NOT NULL, name VARCHAR(3)", allow_casts=["id"])
        df = pd.DataFrame({"id": ["1", "x", None], "name": ["ok", "toolong", "ok"]})
        with self.assertRaises(DataQualityError) as cm:
            TransformationAgent().apply(ctx, df)
        msg = str(cm.exception)
        self.assertIn("cannot be converted to integer", msg)
        self.assertIn("text longer than 3", msg)
        self.assertIn("NULL in a NOT NULL column", msg)
        self.assertTrue(Path(ctx.stats["rejected_file"]).exists())

    def test_quarantine_loads_good_rows_only(self):
        ctx = self.ctx_for("id VARCHAR(5), name VARCHAR(10)", "id INTEGER, name VARCHAR(3)", allow_casts=["id"], on_bad_rows="quarantine")
        df = pd.DataFrame({"id": ["1", "2"], "name": ["ok", "toolong"]})
        out, rejected = TransformationAgent().apply(ctx, df)
        self.assertEqual((len(out), rejected), (1, 1))

    def test_excluded_columns_dropped_and_renamed(self):
        ctx = self.ctx_for("a INTEGER, junk VARCHAR(5), CustID INTEGER", "a INTEGER, CustomerID INTEGER", exclude_columns=["junk"], column_map={"CustID": "CustomerID"})
        out, _ = TransformationAgent().apply(ctx, pd.DataFrame({"a": [1], "junk": ["x"], "CustID": [9]}))
        self.assertEqual(list(out.columns), ["a", "CustomerID"])
        self.assertTrue(any("rename" in s for s in ctx.transform_steps))


# --------------------------------------------------------------------------- 5
EV = "CREATE TABLE Events (EventID INTEGER PRIMARY KEY NOT NULL, Name VARCHAR(20))"
INC = {"source_table": "Events", "dest_table": "Events", "load_type": "incremental", "write_strategy": "append",
       "watermark_column": "EventID", "watermark_type": "integer"}


def make_pipeline(src, dst, control, loader=None, **agents):
    def default_loader(ctx, df):
        df.to_sql(ctx.job_cfg["dest_table"], ctx.dest_engine, if_exists="append", index=False)
        return len(df)

    transformation = TransformationAgent()
    execution = ExecutionAgent(transformation, loader=loader or default_loader, control=control)
    return AgenticPipeline(requirement=RequirementAgent(jobs={"inc": INC}), execution=execution,
                           transformation=transformation, source_engine=src, dest_engine=dst, **agents)


class PipelineTests(unittest.TestCase):
    def test_incremental_moves_only_new_rows_and_advances_watermark(self):
        src, dst = engines([EV], [EV])
        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (1,'a'),(2,'b'),(3,'c')"))
        control = FakeControl()
        pipe = make_pipeline(src, dst, control)

        ctx = pipe.run(job_name="inc")
        self.assertEqual(ctx.status, "success", ctx.error)
        self.assertEqual((ctx.stats["rows_extracted"], ctx.stats["rows_written"]), (3, 3))
        self.assertEqual(control.watermark, "3")

        ctx = pipe.run(job_name="inc")  # nothing new
        self.assertEqual(ctx.stats["rows_extracted"], 0)

        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (4,'d')"))
        ctx = pipe.run(job_name="inc")
        self.assertEqual(ctx.stats["rows_written"], 1)
        self.assertEqual(control.watermark, "4")
        with dst.connect() as c:
            self.assertEqual(c.execute(text("SELECT COUNT(*) FROM Events")).scalar(), 4)

    def test_failed_load_keeps_watermark_and_raises_alert(self):
        src, dst = engines([EV], [EV])
        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (1,'a'),(2,'b')"))
        control = FakeControl()

        def broken_loader(ctx, df):
            raise RuntimeError("Login failed for user 'sa' ... PWD=Secret123;")

        ctx = make_pipeline(src, dst, control, loader=broken_loader).run(job_name="inc")
        self.assertEqual(ctx.status, "failed")
        self.assertIsNone(control.watermark)  # bookmark did not move
        self.assertEqual(control.status, "failed")
        self.assertEqual(ctx.alert["error_class"], "CONNECTION")
        self.assertNotIn("Secret123", json.dumps(ctx.alert))
        self.assertEqual(ctx.error_stage, "Execution Agent")

    def test_validation_failure_stops_before_anything_is_written(self):
        src, dst = engines([EV.replace("Name VARCHAR(20)", "Name VARCHAR(20), Extra INTEGER")], [EV])
        control = FakeControl()
        ctx = make_pipeline(src, dst, control).run(job_name="inc")
        self.assertEqual(ctx.status, "failed")
        self.assertEqual(ctx.error_stage, "Validation Agent")
        self.assertEqual(ctx.alert["error_class"], "VALIDATION_FAILED")
        self.assertIn("NEW_COLUMN_IN_SOURCE", ctx.alert["message"])
        self.assertEqual(control.calls, [])  # execution never started
        self.assertNotIn("rows_extracted", ctx.stats)

    def test_bad_rows_fail_run_and_keep_watermark(self):
        src, dst = engines(["CREATE TABLE Events (EventID INTEGER PRIMARY KEY NOT NULL, Name VARCHAR(20))"],
                           ["CREATE TABLE Events (EventID INTEGER PRIMARY KEY NOT NULL, Name VARCHAR(3))"])
        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (1,'ok'),(2,'far too long')"))
        control = FakeControl()
        ctx = make_pipeline(src, dst, control).run(job_name="inc")
        # length overflow is caught up front by the Validation Agent (VARCHAR(20) -> VARCHAR(3))
        self.assertEqual(ctx.status, "failed")
        self.assertIsNone(control.watermark)

    def test_dry_run_reads_and_writes_nothing(self):
        src, dst = engines([EV], [EV])
        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (1,'a')"))
        control = FakeControl()
        ctx = make_pipeline(src, dst, control).run(job_name="inc", dry_run=True)
        self.assertEqual(ctx.status, "dry_run_ok")
        self.assertEqual(control.calls, [])
        self.assertTrue(ctx.transform_steps)
        with dst.connect() as c:
            self.assertEqual(c.execute(text("SELECT COUNT(*) FROM Events")).scalar(), 0)
            tables = [r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))]
        self.assertNotIn("etl_run_metadata", tables)

    def test_full_reload_resets_watermark(self):
        src, dst = engines([EV], [EV])
        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (1,'a'),(2,'b')"))
        control = FakeControl()
        control.watermark = "2"  # pretend both rows were already loaded
        ctx = make_pipeline(src, dst, control).run(job_name="inc", full_reload=True)
        self.assertEqual(ctx.stats["rows_extracted"], 2)
        self.assertIn("reset", control.calls)

    def test_full_load(self):
        cfg = {"source_table": "Events", "dest_table": "Events", "load_type": "full", "write_strategy": "truncate_load"}
        src, dst = engines([EV], [EV])
        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (1,'a'),(2,'b')"))
        transformation = TransformationAgent()
        pipe = AgenticPipeline(
            requirement=RequirementAgent(jobs={"full": cfg}), transformation=transformation,
            execution=ExecutionAgent(transformation, loader=lambda ctx, df: (df.to_sql("Events", ctx.dest_engine, if_exists="append", index=False), len(df))[1], control=FakeControl()),
            source_engine=src, dest_engine=dst)
        ctx = pipe.run(job_name="full")
        self.assertEqual((ctx.status, ctx.stats["rows_written"]), ("success", 2))


# --------------------------------------------------------------------------- 6
class MetadataTests(unittest.TestCase):
    def test_run_recorded_in_file_and_database(self):
        src, dst = engines([EV], [EV])
        with src.begin() as c:
            c.execute(text("INSERT INTO Events VALUES (1,'a')"))
        ctx = make_pipeline(src, dst, FakeControl()).run(job_name="inc")
        with dst.connect() as c:
            row = c.execute(text("SELECT status, rows_written, watermark_after, job_name FROM etl_run_metadata")).one()
        self.assertEqual(tuple(row), ("success", 1, "1", "inc"))
        last = json.loads(Path(LOG_TMP, "run_metadata.jsonl").read_text().strip().splitlines()[-1])
        self.assertEqual(last["run_id"], ctx.run_id)
        self.assertIn("Validation Agent", last["agent_timings"])

    def test_failed_run_is_recorded_with_error_class(self):
        src, dst = engines([EV.replace("Name VARCHAR(20)", "Name VARCHAR(20), X INTEGER")], [EV])
        make_pipeline(src, dst, FakeControl()).run(job_name="inc")
        with dst.connect() as c:
            row = c.execute(text("SELECT status, error_class, error_stage FROM etl_run_metadata")).one()
        self.assertEqual(tuple(row), ("failed", "VALIDATION_FAILED", "Validation Agent"))

    def test_schema_change_between_runs_is_noticed(self):
        src, dst = engines([EV], [EV])
        pipe = make_pipeline(src, dst, FakeControl())
        pipe.run(job_name="inc")
        with src.begin() as c:
            c.execute(text("ALTER TABLE Events ADD COLUMN Extra INTEGER"))
        with dst.begin() as c:
            c.execute(text("ALTER TABLE Events ADD COLUMN Extra INTEGER"))
        ctx = pipe.run(job_name="inc")
        self.assertIn("SCHEMA_CHANGED_SINCE_LAST_RUN", {f.code for r in ctx.results for f in r.findings})


# --------------------------------------------------------------------------- 7
class ErrorAlertTests(unittest.TestCase):
    def test_classification(self):
        cases = {
            "Login failed for user": "CONNECTION",
            "Invalid object name 'dbo.Nope'": "MISSING_OBJECT",
            "Invalid column name 'X'": "SCHEMA_DRIFT",
            "String or binary data would be truncated": "TRUNCATION",
            "Violation of PRIMARY KEY constraint": "DUPLICATE_KEY",
            "Cannot insert the value NULL into column": "NULL_VIOLATION",
            "Conversion failed when converting": "TYPE_CONVERSION",
            "something odd": "UNKNOWN",
        }
        for msg, expected in cases.items():
            self.assertEqual(classify(RuntimeError(msg)), expected, msg)
        self.assertEqual(classify(DataQualityError("x")), "DATA_QUALITY")

    def test_redaction(self):
        self.assertNotIn("hunter2", redact("DRIVER={x};UID=sa;PWD=hunter2;Encrypt=no"))
        self.assertNotIn("sk-ant-abc123", redact("key sk-ant-abc123"))

    def test_llm_explanation_added_and_failure_tolerated(self):
        ctx = RunContext(job_name="j", error_stage="Execution Agent")
        alert = ErrorAlertAgent(llm=lambda s, p, **k: "The database was unreachable. Start it and retry.").handle(ctx, RuntimeError("Login failed"))
        self.assertIn("unreachable", alert["explanation"])

        def boom(*a, **k):
            raise RuntimeError("api down")

        self.assertNotIn("explanation", ErrorAlertAgent(llm=boom).handle(ctx, RuntimeError("Login failed")))

    def test_webhook_delivery(self):
        received = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.handle_request, daemon=True).start()
        os.environ["ETL_ALERT_WEBHOOK_URL"] = f"http://127.0.0.1:{server.server_port}/hook"
        try:
            alert = ErrorAlertAgent(llm=lambda *a, **k: None).handle(RunContext(job_name="j", error_stage="Execution Agent"), RuntimeError("Login failed"))
        finally:
            del os.environ["ETL_ALERT_WEBHOOK_URL"]
            server.server_close()
        self.assertIn("webhook", alert["delivered_to"])
        self.assertIn("ETL ALERT [CONNECTION]", received[0]["text"])

    def test_email_delivery_with_fake_smtp(self):
        sent = []

        class FakeSMTP:
            def __init__(self, host, port, timeout=0): sent.append(("connect", host, port))
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def starttls(self): sent.append("tls")
            def login(self, u, p): sent.append(("login", u))
            def send_message(self, msg): sent.append(("msg", msg["To"], msg["Subject"]))

        original = error_alert_agent.smtplib.SMTP
        error_alert_agent.smtplib.SMTP = FakeSMTP
        os.environ.update(ETL_ALERT_EMAIL_TO="me@example.com", ETL_SMTP_HOST="smtp.example.com", ETL_SMTP_USER="u", ETL_SMTP_PASSWORD="p")
        try:
            alert = ErrorAlertAgent(llm=lambda *a, **k: None).handle(RunContext(job_name="j", error_stage="x"), RuntimeError("boom"))
        finally:
            error_alert_agent.smtplib.SMTP = original
            for k in ("ETL_ALERT_EMAIL_TO", "ETL_SMTP_HOST", "ETL_SMTP_USER", "ETL_SMTP_PASSWORD"):
                os.environ.pop(k, None)
        self.assertIn("email", alert["delivered_to"])
        self.assertIn(("msg", "me@example.com", "[ETL ALERT] j: UNKNOWN"), sent)

    def test_alert_written_to_file(self):
        ErrorAlertAgent(llm=lambda *a, **k: None).handle(RunContext(job_name="filejob", error_stage="x"), RuntimeError("boom"))
        self.assertIn("filejob", Path(LOG_TMP, "alerts.log").read_text())


if __name__ == "__main__":
    unittest.main()
