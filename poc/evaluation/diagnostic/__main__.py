from __future__ import annotations

import argparse

from poc.evaluation.diagnostic.cli import add_arguments, run_cli
from poc.evaluation.suite.__main__ import load_environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast infrastructure diagnostics; no performance benchmark"
    )
    add_arguments(parser)
    args = parser.parse_args()
    try:
        if args.live_config is not None:
            load_environment()
        status = run_cli(args)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
