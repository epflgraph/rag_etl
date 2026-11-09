import argparse
import logging
import sys

from dotenv import find_dotenv, load_dotenv

from rag_etl.projects import from_code
from rag_etl.pipeline import CACHED_STEPS, run_pipeline


class _DropMediaUploadErrors(logging.Filter):
    """Langfuse media uploads can fail without affecting the run; they are noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Media upload error" not in record.getMessage()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-etl",
        description="RAG ETL pipeline: extract, transform and load material into the RAG database.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run the pipeline for a project.")
    run.add_argument("--project", required=True, help="Project code, e.g. COM202 (as in the class name).")
    run.add_argument("--dry", action="store_true", help="Dry run: process resources without publishing anything.")
    run.add_argument("--refresh", choices=CACHED_STEPS, help="Reprocess the named step, bypassing its cache.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # The settings and secrets come from the environment; a .env found from
    # the working directory upwards fills in what the environment lacks
    load_dotenv(find_dotenv(usecwd=True))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("langfuse").addFilter(_DropMediaUploadErrors())

    if args.command == "run":
        logging.info(f"Run configuration: project={args.project}, dry={args.dry}, refresh={args.refresh or 'none'}")

        try:
            spec = from_code(args.project)
        except ValueError as e:
            parser.error(str(e))

        run_pipeline(spec, dry=args.dry, refresh=args.refresh)


if __name__ == "__main__":
    main()
