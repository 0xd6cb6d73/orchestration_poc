"""Durable report utilities and comparisons over matched recorded trials."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO, cast
from uuid import uuid4


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def trial_key(trial: dict[str, Any]) -> str:
    return json.dumps(
        [trial["case_id"], trial["model_name"], trial["strategy"], trial["repetition"]],
        separators=(",", ":"),
    )


@contextmanager
def locked(path: Path) -> Generator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process is writing {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def append(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def records(path: Path, *, repair: bool = False) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    result: list[dict[str, Any]] = []
    offset = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("report record must be an object")
            result.append(cast(dict[str, Any], record))
        except (ValueError, UnicodeDecodeError):
            # Only an unterminated last record can be an interrupted write.
            if index != len(lines) - 1 or line.endswith(b"\n") or not result:
                raise ValueError(f"corrupt report record {index + 1} in {path}") from None
            if repair:
                backup = path.with_suffix(path.suffix + ".interrupted")
                if backup.exists():
                    backup = path.with_suffix(path.suffix + f".interrupted-{uuid4().hex}")
                with backup.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
            break
        offset += len(line)
    if repair and offset == len(raw) and raw and not raw.endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return result


def coverage(report: dict[str, Any]) -> dict[str, Any]:
    config = report["config"]
    expected = (
        len(report["cases"])
        * len(config["models"])
        * len(config["strategies"])
        * config["repetitions"]
    )
    keys = [trial_key(t) for t in report["trials"]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate terminal trial records")
    return {
        "expected": expected,
        "recorded": len(keys),
        "missing": expected - len(keys),
        "complete": len(keys) == expected,
    }


def comparisons(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, dict[tuple[str, int], dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for t in trials:
        groups[t["model_name"], t["family"]][t["strategy"]][t["case_id"], t["repetition"]] = t
    result: list[dict[str, Any]] = []
    for (model, family), strategies in sorted(groups.items()):
        for left, right in [("single", "review"), ("single-json", "review-json")]:
            a, b = strategies.get(left, {}), strategies.get(right, {})
            keys = a.keys() & b.keys()
            result.append(
                {
                    "model_name": model,
                    "family": family,
                    "strategies": [left, right],
                    "matched_trials": len(keys),
                    "matched_cases": len({k[0] for k in keys}),
                    "unmatched_trials": len(a.keys() ^ b.keys()),
                    "pass_rate_delta": sum(
                        b[k]["scores"]["exact"] - a[k]["scores"]["exact"] for k in keys
                    )
                    / len(keys)
                    if keys
                    else None,
                    "mean_seconds_delta": sum(
                        b[k]["elapsed_seconds"] - a[k]["elapsed_seconds"] for k in keys
                    )
                    / len(keys)
                    if keys
                    else None,
                }
            )
    return result
