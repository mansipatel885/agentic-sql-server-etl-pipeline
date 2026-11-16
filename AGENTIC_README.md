# Agentic ETL add-on

Adds seven cooperating agents on top of your existing SQL Server pipeline.
Your files (config.py, full_load.py, incremental_load.py, ...) are NOT changed.

| # | Agent | Job |
|---|-------|-----|
| a | Requirement Agent | Reads the job (or a plain-English request) and checks it is complete and consistent |
| b | Validation Agent | Compares source and destination: schema, data types, lengths, precision, keys, watermark |
| c | Mapping Agent | Builds the source-to-destination column map (renames, exclusions) |
| d | Transformation Agent | Trims, casts and checks every row; bad rows fail the run (or are quarantined) |
| e | Execution Agent | Runs the load with your existing full/incremental code; moves the watermark only on success |
| f | Metadata Agent | Records every run (rows, timings, schema fingerprint, status) to `etl_run_metadata` and `logs/run_metadata.jsonl` |
| g | Error and Alert Agent | Classifies the failure, says what to do next, writes `logs/alerts.log`, optional webhook/email |

Claude (optional, needs `ANTHROPIC_API_KEY` and `pip install anthropic`) is used only to
propose a job from plain English and to explain errors. Deterministic agents verify
everything it proposes, and nothing is written unless every check passes.

## Install
Unzip into your etl_project folder (merge/keep existing files), then:

    source .venv/bin/activate
    pip install -r requirements-agentic.txt     # only needed for the optional Claude features

## Run (SQL Server container must be running)
    python run_agentic.py --list
    python run_agentic.py --job <job_name> --dry-run     # checks everything, writes nothing
    python run_agentic.py --job <job_name>
    python run_agentic.py --job <job_name> --full-reload

## What has and has not been tested
- Tested: 44 unit tests on SQLite stand-ins (all agents, failure paths, dry run, alerts via a local webhook and a fake SMTP server).
  Run them with: `python -m unittest discover -s tests`
- NOT yet tested: a run against your Docker SQL Server and the real Claude API.
  Do `--dry-run` first, then a real run, before describing it as tested anywhere.
