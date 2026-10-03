from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["file_sha256", "write_csv"]
