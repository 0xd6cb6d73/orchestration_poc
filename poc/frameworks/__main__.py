"""Run one headless native-framework request from JSON."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from poc.frameworks.app import RunRequest, start


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one native orchestration candidate")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-env-file", action="store_true")
    args = parser.parse_args()
    if not args.no_env_file:
        load_dotenv(Path.cwd() / ".env", override=False, interpolate=False)
    request = RunRequest.model_validate_json(args.request.read_bytes())

    async def execute() -> int:
        result = await start(request, args.output).result()
        print(json.dumps(result.model_dump(mode="json"), indent=2))
        return 0 if result.status == "succeeded" else 1

    raise SystemExit(asyncio.run(execute()))


if __name__ == "__main__":
    main()
