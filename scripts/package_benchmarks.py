"""Build independent benchmark ZIPs, splitting only between whole examples.

Run with --source LOCAL_THERMOML_OUTPUTS --destination NEW_STAGING_DIRECTORY.
Source paths and bytes are preserved. Campaign files repeat in every part.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

LIMIT = 100_000_000
EXAMPLE = re.compile(r"^Q\d")
CHUNK = 1024 * 1024


def natural(value):
    return tuple((1, int(x)) if x.isdigit() else (0, x.lower())
                 for x in re.split(r"(\d+)", str(value)))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root):
    groups, shared, fingerprints = {}, [], {}
    for path in sorted(root.rglob("*"), key=lambda p: natural(p.relative_to(root).as_posix())):
        if path.is_symlink():
            raise ValueError(f"Symlink is not a saved output file: {path}")
        relative = path.relative_to(root)
        # Only directories can be example boundaries.
        dirs = relative.parts if path.is_dir() else relative.parts[:-1]
        boundary = next((i for i, part in enumerate(dirs) if EXAMPLE.match(part)), None)
        key = "/".join(dirs[:boundary + 1]) if boundary is not None else None
        if key is not None:
            groups.setdefault(key, [])
        # Preserve empty directories explicitly.
        if path.is_dir() and any(path.iterdir()):
            continue
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"Unsupported source entry: {path}")
        name = relative.as_posix() + ("/" if path.is_dir() else "")
        (groups[key] if key is not None else shared).append((name, path))
        if path.is_file():
            stat = path.stat()
            fingerprints[name] = (stat.st_size, stat.st_mtime_ns)
    if not groups:
        raise ValueError(f"No example folders found: {root}")
    return groups, shared, fingerprints


def snapshot(entries, target, hashes):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in entries:
            if name.endswith("/"):
                archive.write(path, name)
                continue
            before = path.stat()
            digest = hashlib.sha256()
            info = zipfile.ZipInfo.from_file(path, name, strict_timestamps=False)
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w") as dest:
                for chunk in iter(lambda: source.read(CHUNK), b""):
                    dest.write(chunk)
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed during snapshot: {path}")
            hashes[name] = digest.hexdigest()
    # All collections are far below ZIP64 entry-count/size thresholds.
    with zipfile.ZipFile(target) as archive:
        if len(archive.infolist()) >= 65535:
            raise ValueError("ZIP64 entry counts are not supported by this packer")
    return target.stat().st_size - 22


def partitions(examples, sizes, shared_size, limit):
    result, current, size = [], [], shared_size + 22
    for example in sorted(examples, key=natural):
        required = sizes[example]
        if required + shared_size + 22 > limit:
            raise ValueError(f"One complete example exceeds ZIP limit: {example}")
        if current and size + required > limit:
            result.append(current)
            current, size = [], shared_size + 22
        current.append(example)
        size += required
    if current:
        result.append(current)
    return result


def range_token(example):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", example)


def build_collection(source_root, relative, dest_relative, destination, limit):
    root = source_root / relative
    examples, shared, before = inventory(root)
    hashes, sizes, snapshots, records = {}, {}, {}, []
    target_dir = destination / dest_relative.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Snapshot {relative}: {len(examples)} examples, {len(before)} files", flush=True)
    with tempfile.TemporaryDirectory(prefix="thermoml-zips-") as tmp:
        temp = Path(tmp)
        shared_zip = temp / "shared.zip"
        shared_size = snapshot(shared, shared_zip, hashes)
        for index, (example, entries) in enumerate(examples.items()):
            snapshots[example] = temp / f"example-{index:04d}.zip"
            sizes[example] = snapshot(entries, snapshots[example], hashes)
        groups = partitions(examples, sizes, shared_size, limit)
        for index, group in enumerate(groups, 1):
            suffix = "" if len(groups) == 1 else (
                f"__part{index:02d}__{range_token(group[0])}--{range_token(group[-1])}")
            target = target_dir / f"{dest_relative.name}{suffix}.zip"
            expected = {name for name, _ in shared}
            for example in group:
                expected.update(name for name, _ in examples[example])
            if len(expected) >= 65535:
                raise ValueError("ZIP64 entry counts are not supported by this packer")
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
                for snap in [shared_zip] + [snapshots[example] for example in group]:
                    with zipfile.ZipFile(snap) as source:
                        for original in source.infolist():
                            info = copy.copy(original)
                            with source.open(info.filename) as read, output.open(info, "w") as write:
                                shutil.copyfileobj(read, write, CHUNK)
            if target.stat().st_size > limit:
                raise ValueError(f"Final archive exceeds limit: {target}")
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != expected:
                    raise ValueError(f"Archive entry mismatch: {target}")
                for name in names:
                    if name.endswith("/"):
                        continue
                    with archive.open(name) as handle:
                        if hashlib.file_digest(handle, "sha256").hexdigest() != hashes[name]:
                            raise ValueError(f"Archive byte mismatch: {target}: {name}")
            record = {
                "path": target.relative_to(destination).as_posix(),
                "source_collection": relative.as_posix(),
                "part": index, "parts": len(groups),
                "first_example": group[0], "last_example": group[-1],
                "examples": group,
                "shared_members": [name for name, _ in shared],
                "file_count": sum(not name.endswith("/") for name in expected),
                "bytes": target.stat().st_size, "sha256": sha256(target),
            }
            records.append(record)
            print(f"Verified {record['path']}: {record['bytes']:,} bytes", flush=True)
    after_examples, after_shared, after = inventory(root)
    if before != after or list(examples) != list(after_examples) or shared != after_shared:
        raise ValueError(f"Source collection changed during build: {root}")
    return records


def collections(source):
    result = []
    for agent in ("Main", "Query", "Analysis"):
        for mode in ("test_run_full_capability", "test_run_ablation"):
            root = source / agent / mode
            if not root.is_dir():
                raise ValueError(f"Missing source mode: {root}")
            if mode == "test_run_full_capability" and agent != "Query":
                result.append((Path(agent) / mode, Path(agent) / mode / mode))
            else:
                for path in sorted(root.iterdir(), key=lambda p: natural(p.name)):
                    if not path.is_dir():
                        raise ValueError(f"Unexpected file outside a collection: {path}")
                    relative = path.relative_to(source)
                    result.append((relative, relative))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True,
                        help="New staging directory; existing directories are refused")
    parser.add_argument("--max-bytes", type=int, default=LIMIT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 0 < args.max_bytes <= LIMIT:
        parser.error(f"--max-bytes must be between 1 and {LIMIT}")
    source = args.source.absolute()
    plan = collections(source)
    args.destination.mkdir(parents=True, exist_ok=False)

    def run(item):
        return build_collection(source, item[0], item[1], args.destination, args.max_bytes)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        archives = [record for result in pool.map(run, plan) for record in result]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "_Code_Maintenance/_outputs_user/Benchmark/ThermoML",
        "max_archive_bytes": args.max_bytes,
        "compression": "ZIP_DEFLATED",
        "split_policy": "Whole example directories in natural path order; shared campaign files repeated in every part",
        "archives": archives,
    }
    (args.destination / "ARCHIVES.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: {len(archives)} verified archives; manifest {args.destination / 'ARCHIVES.json'}", flush=True)


if __name__ == "__main__":
    main()
