"""Restore ordinary Git-tracked database ZIP parts to their original paths.

Run ``python database_archives.py`` once, or use the browser launcher. A checked
receipt makes later launches stat-only; no database/archive payload is reread.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import tempfile
import time
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parent
ARCHIVE_LIMIT = 100_000_000
BUFFER_BYTES = 1024 * 1024
SCHEMA_VERSION = 1


class DatabaseArchiveError(RuntimeError):
    """A database cannot be restored safely from its published archive parts."""


def _relative_path(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or "\x00" in value or value.startswith("/")
            or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise DatabaseArchiveError(f"Unsafe relative path: {value!r}")
    return PurePosixPath(value)


def _confined_path(root: Path, relative: str) -> Path:
    """Reject traversal and symlinks/junctions, including existing ancestors."""
    parts = _relative_path(relative).parts
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)():
            raise DatabaseArchiveError(f"Symlink/junction is not an archive/cache path: {candidate}")
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise DatabaseArchiveError(f"Path leaves repository: {candidate}") from exc
    return candidate


def _integer(value, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DatabaseArchiveError(f"Invalid {field}: {value!r}")
    return value


def _digest(value, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise DatabaseArchiveError(f"Invalid SHA256 in {field}")
    return value


def _fingerprint(path: Path) -> dict:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise DatabaseArchiveError(f"Expected a regular file: {path}")
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns, "inode": info.st_ino}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(root: Path, manifest_path: Path | str | None):
    path = Path(manifest_path) if manifest_path is not None else root / "_database_archives" / "manifest.json"
    if not path.is_absolute():
        path = root / path
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise DatabaseArchiveError("Database manifest must be inside the repository") from exc
    path = _confined_path(root, relative)
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            raise DatabaseArchiveError("Database archive manifest is unexpectedly large")
        payload = path.read_bytes()
        doc = json.loads(payload)
    except (OSError, ValueError) as exc:
        raise DatabaseArchiveError(f"Cannot read database archive manifest: {path}: {exc}") from exc
    if (not isinstance(doc, dict) or doc.get("schema_version") != SCHEMA_VERSION
            or not isinstance(doc.get("databases"), list) or not doc["databases"]):
        raise DatabaseArchiveError("Unsupported or empty database archive manifest")
    paths, archives = set(), set()
    for database in doc["databases"]:
        if not isinstance(database, dict):
            raise DatabaseArchiveError("Invalid database record")
        relative = str(_relative_path(database.get("repository_path")))
        if not relative.lower().endswith(".db") or relative.startswith((".database_cache/", "_database_archives/")):
            raise DatabaseArchiveError(f"Invalid database destination: {relative}")
        if relative.casefold() in paths:
            raise DatabaseArchiveError(f"Duplicate database destination: {relative}")
        paths.add(relative.casefold())
        _confined_path(root, relative)
        total = _integer(database.get("bytes"), "database bytes", 1)
        _digest(database.get("sha256"), relative)
        parts = database.get("parts")
        if not isinstance(parts, list) or not parts:
            raise DatabaseArchiveError(f"No archive parts for {relative}")
        offset = 0
        for part in parts:
            if not isinstance(part, dict):
                raise DatabaseArchiveError(f"Invalid archive part for {relative}")
            archive_name = str(_relative_path(part.get("archive")))
            _relative_path(part.get("member"))
            if not archive_name.lower().endswith(".zip") or archive_name.casefold() in archives:
                raise DatabaseArchiveError(f"Invalid or duplicate archive: {archive_name}")
            archives.add(archive_name.casefold())
            _confined_path(path.parent, archive_name)
            if _integer(part.get("offset"), "part offset") != offset:
                raise DatabaseArchiveError(f"Noncontiguous or out-of-order archive parts for {relative}")
            offset += _integer(part.get("bytes"), "part bytes", 1)
            archive_size = _integer(part.get("archive_bytes"), "archive bytes", 1)
            if archive_size > ARCHIVE_LIMIT:
                raise DatabaseArchiveError(f"Archive exceeds {ARCHIVE_LIMIT} bytes: {archive_name}")
            _digest(part.get("sha256"), "member sha256")
            _digest(part.get("archive_sha256"), "archive sha256")
        if offset != total:
            raise DatabaseArchiveError(f"Archive part lengths do not match {relative}")
    return doc, path.parent, hashlib.sha256(payload).hexdigest()


@contextmanager
def _preparation_lock(cache: Path, timeout: float):
    """An OS-owned lock is released automatically if a preparing process exits."""
    lock_path = _confined_path(cache, "prepare.lock")
    with lock_path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise DatabaseArchiveError("Timed out waiting for database preparation") from exc
                time.sleep(0.05)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _archive_fingerprints(doc: dict, archive_root: Path) -> dict:
    fingerprints = {}
    for database in doc["databases"]:
        for part in database["parts"]:
            path = _confined_path(archive_root, part["archive"])
            try:
                stamp = _fingerprint(path)
            except OSError as exc:
                raise DatabaseArchiveError(f"Missing/unreadable database archive: {path}") from exc
            if stamp["bytes"] != part["archive_bytes"]:
                raise DatabaseArchiveError(f"Archive size mismatch: {path}")
            fingerprints[part["archive"]] = stamp
    return fingerprints


def _restore_database(root: Path, archive_root: Path, database: dict) -> Path:
    target = _confined_path(root, database["repository_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    _confined_path(root, database["repository_path"])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".assembling",
                                         dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            whole_digest = hashlib.sha256()
            total = 0
            for part in database["parts"]:
                archive_path = _confined_path(archive_root, part["archive"])
                if _sha256(archive_path) != part["archive_sha256"]:
                    raise DatabaseArchiveError(f"Archive checksum mismatch: {archive_path}")
                with zipfile.ZipFile(archive_path) as archive:
                    entries = archive.infolist()
                    if len(entries) != 1:
                        raise DatabaseArchiveError(f"Expected exactly one member in {archive_path}")
                    info = entries[0]
                    _relative_path(info.orig_filename)
                    if (info.orig_filename != part["member"] or info.is_dir()
                            or stat.S_ISLNK(info.external_attr >> 16)
                            or info.file_size != part["bytes"]):
                        raise DatabaseArchiveError(f"Unexpected archive member in {archive_path}")
                    digest, length = hashlib.sha256(), 0
                    with archive.open(info) as source:
                        for block in iter(lambda: source.read(BUFFER_BYTES), b""):
                            length += len(block)
                            if length > part["bytes"]:
                                raise DatabaseArchiveError(f"Oversized archive member in {archive_path}")
                            output.write(block)
                            digest.update(block)
                            whole_digest.update(block)
                    if length != part["bytes"] or digest.hexdigest() != part["sha256"]:
                        raise DatabaseArchiveError(f"Member checksum/size mismatch: {archive_path}")
                    total += length
            output.flush()
            os.fsync(output.fileno())
            if total != database["bytes"] or whole_digest.hexdigest() != database["sha256"]:
                raise DatabaseArchiveError(f"Reassembled database checksum/size mismatch: {target}")
        target = _confined_path(root, database["repository_path"])
        if target.exists():
            raise DatabaseArchiveError(f"Database appeared during preparation; preserved: {target}")
        # Publish without replacing a database created outside our lock.
        if os.name == "nt":
            os.rename(temporary, target)  # Windows rename refuses an existing target.
        else:
            os.link(temporary, target)  # Atomic exclusive creation on the same filesystem.
            temporary.unlink()
        temporary = None
        return target
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, DatabaseArchiveError):
            raise
        raise DatabaseArchiveError(f"Could not restore {target}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_receipt(path: Path, receipt: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".receipt-",
                                         dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(receipt, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_databases(root: Path | str | None = None, *, manifest_path: Path | str | None = None,
                     lock_timeout: float = 120.0, progress=None) -> list[dict]:
    """Restore missing databases, adopt matching originals, and reuse checked files.

    Existing databases with different bytes are preserved and reported as errors.
    The receipt is an optimization, not a data source; changed files are checked
    again. No network, SQLite connection, extraction directory or schema change.
    """
    root = Path(root or REPOSITORY_ROOT).resolve()
    cache = _confined_path(root, ".database_cache")
    cache.mkdir(exist_ok=True)
    _confined_path(root, ".database_cache")
    receipt_path = _confined_path(cache, "ready.json")
    with _preparation_lock(cache, lock_timeout):
        # Read again after waiting: a concurrent publisher may have changed it.
        try:
            doc, archive_root, manifest_sha = _read_manifest(root, manifest_path)
        except Exception:
            receipt_path.unlink(missing_ok=True)
            raise
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            receipt = {}
        if not isinstance(receipt, dict):
            receipt = {}
        try:
            archive_stamps = _archive_fingerprints(doc, archive_root)
            targets = {}
            for database in doc["databases"]:
                target = _confined_path(root, database["repository_path"])
                if target.exists():
                    targets[database["repository_path"]] = _fingerprint(target)
            cached_sources_match = (
                receipt.get("manifest_sha256") == manifest_sha
                and receipt.get("root") == str(root)
                and receipt.get("archives") == archive_stamps
                and isinstance(receipt.get("targets"), dict)
            )
            if (cached_sources_match and len(targets) == len(doc["databases"])
                    and receipt["targets"] == targets):
                return [{"repository_path": item["repository_path"], "status": "reused"}
                        for item in doc["databases"]]
            receipt_path.unlink(missing_ok=True)
            results = []
            for database in doc["databases"]:
                target = _confined_path(root, database["repository_path"])
                if target.exists():
                    before = _fingerprint(target)
                    if (cached_sources_match
                            and receipt["targets"].get(database["repository_path"]) == before):
                        # A missing sibling does not invalidate this database's
                        # previous verification while its sources and stat agree.
                        status = "reused"
                    else:
                        if before["bytes"] != database["bytes"]:
                            raise DatabaseArchiveError(
                                f"Existing database differs from the published bytes; preserved: {target}. "
                                "Move it aside before preparing the published database.")
                        # Adoption verifies packed sources as well as the existing DB.
                        for part in database["parts"]:
                            archive_path = _confined_path(archive_root, part["archive"])
                            if _sha256(archive_path) != part["archive_sha256"]:
                                raise DatabaseArchiveError(f"Archive checksum mismatch: {archive_path}")
                        if _sha256(target) != database["sha256"]:
                            raise DatabaseArchiveError(
                                f"Existing database differs from the published bytes; preserved: {target}. "
                                "Move it aside before preparing the published database.")
                        if _fingerprint(target) != before:
                            raise DatabaseArchiveError(f"Database changed during verification: {target}")
                        status = "verified"
                else:
                    if progress:
                        progress(f"Preparing {database['repository_path']} ({database['bytes']:,} bytes)")
                    target = _restore_database(root, archive_root, database)
                    status = "restored"
                targets[database["repository_path"]] = _fingerprint(target)
                results.append({"repository_path": database["repository_path"], "status": status})
            if _archive_fingerprints(doc, archive_root) != archive_stamps:
                raise DatabaseArchiveError("Database archives changed during preparation")
            _write_receipt(receipt_path, {"schema_version": SCHEMA_VERSION, "root": str(root),
                           "manifest_sha256": manifest_sha, "archives": archive_stamps,
                           "targets": targets})
            return results
        except Exception:
            receipt_path.unlink(missing_ok=True)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        results = ensure_databases(args.root, manifest_path=args.manifest,
                                   progress=lambda message: print(message, flush=True))
    except (DatabaseArchiveError, OSError) as exc:
        print(f"Database preparation failed: {exc}", flush=True)
        return 1
    print("Databases ready: " + ", ".join(f"{row['repository_path']} ({row['status']})" for row in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
