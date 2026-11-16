"""
Command line for the agentic pipeline.

    python run_agentic.py --list
    python run_agentic.py --job customers_full --dry-run     # check everything, write nothing
    python run_agentic.py --job orders_incremental           # real run
    python run_agentic.py --job orders_incremental --full-reload
    python run_agentic.py --request "load new orders into dest, upsert on OrderID"   # needs ANTHROPIC_API_KEY
"""

import argparse
import logging
import sys

from pipeline import AgenticPipeline, format_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an ETL job through the agents.")
    parser.add_argument("--job", help="job name from config.JOBS")
    parser.add_argument("--request", help="plain-English request (uses Claude to propose the job)")
    parser.add_argument("--dry-run", action="store_true", help="run every check, read and write nothing")
    parser.add_argument("--full-reload", action="store_true", help="incremental jobs: reset the watermark first")
    parser.add_argument("--list", action="store_true", help="list configured jobs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.list:
        from config import JOBS

        for name, cfg in JOBS.items():
            print(f"  - {name}  ({cfg['load_type']}, {cfg['write_strategy']})")
        return 0
    if not args.job and not args.request:
        parser.error("give --job or --request (or --list)")

    ctx = AgenticPipeline().run(
        job_name=args.job, request_text=args.request, full_reload=args.full_reload, dry_run=args.dry_run
    )
    print(format_report(ctx))
    return 0 if ctx.status in ("success", "dry_run_ok") else 1


if __name__ == "__main__":
    sys.exit(main())
