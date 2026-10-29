"""Package exact SQLite database bytes into independent Git-sized ZIP parts.

Defaults use this checkout's docs/RELEASE_ASSETS.json even when --source-root
points to a different local checkout. Destination must be a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database_archives import (
    ARCHIVE_LIMIT, BUFFER_BYTES, DatabaseArchiveError, _confined_path,
    _digest, _fingerprint, _integer, _relative_path, _sha256,
)

DEFAULT_CHUNK_BYTES = 95_000_000


def build_archives(source_root: Path | str, destination: Path | str,
                   inventory: Path | str | dict | None = None, *,
                   chunk_bytes: int = DEFAULT_CHUNK_BYTES,
                   compression: str = "deflated", progress=print) -> dict:
    """Stream each original once; publish a manifest only after all hashes match."""
    source_root = Path(source_root).resolve()
    destination = Path(destination).absolute()
    chunk_bytes = _integer(chunk_bytes, "chunk bytes", 1)
    if chunk_bytes > DEFAULT_CHUNK_BYTES:
        raise DatabaseArchiveError(f"chunk_bytes must be <= {DEFAULT_CHUNK_BYTES}")
    if compression not in {"deflated", "stored"}:
        raise DatabaseArchiveError("compression must be deflated or stored")
    if isinstance(inventory, dict):
        document = inventory
    else:
        inventory = Path(inventory) if inventory is not None else REPO_ROOT / "docs" / "RELEASE_ASSETS.json"
        document = json.loads(inventory.read_text(encoding="utf-8"))
    assets = document.get("assets") if isinstance(document, dict) else None
    if not isinstance(assets, list) or not assets:
        raise DatabaseArchiveError("Inventory must contain a nonempty assets list")
    source_records, seen, archive_stems = [], set(), set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise DatabaseArchiveError("Invalid database inventory record")
        relative = str(_relative_path(asset.get("repository_path")))
        if not relative.lower().endswith(".db") or relative.casefold() in seen:
            raise DatabaseArchiveError(f"Duplicate/invalid database destination: {relative}")
        seen.add(relative.casefold())
        source = _confined_path(source_root, relative)
        if source.stem.casefold() in archive_stems:
            raise DatabaseArchiveError(f"Database basenames must be unique: {source.name}")
        archive_stems.add(source.stem.casefold())
        expected_bytes = _integer(asset.get("bytes"), "database bytes", 1)
        expected_sha = _digest(asset.get("sha256"), relative)
        before = _fingerprint(source)
        if before["bytes"] != expected_bytes:
            raise DatabaseArchiveError(f"Source size disagrees with inventory: {source}")
        source_records.append((relative, source, expected_bytes, expected_sha, before))
    if destination.exists() or destination.is_symlink():
        raise DatabaseArchiveError(f"Destination must be a new directory: {destination}")
    # The destination's existing parents must not redirect writes through links.
    _confined_path(Path(destination.anchor), destination.relative_to(destination.anchor).as_posix())
    destination.mkdir(parents=True)
    manifest = {"schema_version": 1, "archive_limit_bytes": ARCHIVE_LIMIT,
                "chunk_bytes": chunk_bytes, "compression": compression,
                "compression_level": 1 if compression == "deflated" else None,
                "databases": []}
    method = zipfile.ZIP_DEFLATED if compression == "deflated" else zipfile.ZIP_STORED
    for relative, source, expected_bytes, expected_sha, before in source_records:
        record = {"repository_path": relative, "bytes": expected_bytes,
                  "sha256": expected_sha, "parts": []}
        whole_digest = hashlib.sha256()
        offset = 0
        with source.open("rb") as original:
            part_number = 0
            while offset < expected_bytes:
                part_number += 1
                length = min(chunk_bytes, expected_bytes - offset)
                archive_name = f"{source.stem}_part{part_number}.zip"
                member_name = f"part{part_number}.bin"
                archive_path = destination / archive_name
                digest = hashlib.sha256()
                with zipfile.ZipFile(archive_path, "w", compression=method, compresslevel=1,
                                     allowZip64=False) as archive:
                    with archive.open(member_name, "w") as member:
                        remaining = length
                        while remaining:
                            block = original.read(min(BUFFER_BYTES, remaining))
                            if not block:
                                raise DatabaseArchiveError(f"Source ended during packaging: {source}")
                            member.write(block)
                            digest.update(block)
                            whole_digest.update(block)
                            remaining -= len(block)
                archive_bytes = archive_path.stat().st_size
                if archive_bytes > ARCHIVE_LIMIT:
                    raise DatabaseArchiveError(f"ZIP exceeds Git file limit: {archive_path}")
                record["parts"].append({"archive": archive_name, "member": member_name,
                    "offset": offset, "bytes": length, "sha256": digest.hexdigest(),
                    "archive_bytes": archive_bytes, "archive_sha256": _sha256(archive_path)})
                offset += length
                if progress:
                    progress(f"Packed {archive_name}: {length:,} database bytes -> {archive_bytes:,} ZIP bytes")
            if original.read(1):
                raise DatabaseArchiveError(f"Source grew during packaging: {source}")
        if _fingerprint(source) != before:
            raise DatabaseArchiveError(f"Source changed during packaging: {source}")
        if whole_digest.hexdigest() != expected_sha:
            raise DatabaseArchiveError(f"Source SHA256 disagrees with inventory: {source}")
        manifest["databases"].append(record)
        if progress:
            progress(f"Verified original SHA256: {relative}")
    temporary = destination / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination / "manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=REPO_ROOT / "docs" / "RELEASE_ASSETS.json")
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--compression", choices=("deflated", "stored"), default="deflated")
    args = parser.parse_args(argv)
    try:
        manifest = build_archives(args.source_root, args.destination, args.inventory,
                                  chunk_bytes=args.chunk_bytes, compression=args.compression,
                                  progress=lambda message: print(message, flush=True))
    except (DatabaseArchiveError, OSError, ValueError) as exc:
        print(f"Database packaging failed: {exc}", file=sys.stderr)
        return 1
    print(f"Published manifest for {len(manifest['databases'])} databases in {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
