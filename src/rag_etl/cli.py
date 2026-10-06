import argparse
import logging
import sys

from rag_etl.projects import BaseProject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-etl",
        description="RAG ETL pipeline: extract, transform and load material into the RAG database.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run the pipeline for a project.")
    run.add_argument("--project", required=True, help="Project code, e.g. COM202 (as in the class name).")
    run.add_argument("--dry", action="store_true", help="Dry run: process resources without publishing anything.")
    run.add_argument("--refresh", metavar="TRANSFORM", help="Reprocess the named transform, bypassing its cache.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.command == "run":
        logging.info(f"Run configuration: project={args.project}, dry={args.dry}, refresh={args.refresh or 'none'}")

        try:
            project = BaseProject.from_code(args.project)
        except ValueError as e:
            parser.error(str(e))

        project.run()


if __name__ == "__main__":
    main()
