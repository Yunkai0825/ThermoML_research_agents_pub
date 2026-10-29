"""Package saved statistics as deterministic, independently readable ZIP parts."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import stat
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_stats_archives import ARCHIVE_LIMIT, _safe_path


def _zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def package(source, destination, max_bytes=ARCHIVE_LIMIT):
    source, destination = Path(source), Path(destination)
    if not 0 < max_bytes <= ARCHIVE_LIMIT:
        raise ValueError(f'max_bytes must be between 1 and {ARCHIVE_LIMIT}')
    if destination.exists():
        raise ValueError('Destination must be a new staging directory')
    records = []
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source).as_posix()
        if '/' not in relative and (relative == 'manifest.json' or re.fullmatch(r'eval_stats(?:__part[0-9]{2,})?\.zip', relative)):
            continue
        if any(part in {'__pycache__', '.pytest_cache', '__obsolete__', 'raw_data'} for part in path.relative_to(source).parts):
            continue
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError(f'Symlink/junction is not a payload file: {path}')
        if not path.is_file():
            continue
        parts = relative.split('/')
        if not ((len(parts) == 1 and relative in {'eval_stats.py', 'selection.json'})
                or (len(parts) > 1 and parts[0] in {'eval_workflow', 'numeric_outcomes'})):
            raise ValueError(f'Unexpected payload file: {relative}')
        _safe_path(source, relative)
        data = path.read_bytes()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(_zip_info(relative), data)
        records.append((relative, data, len(buffer.getvalue()) - 22))
    if not records:
        raise ValueError('Statistics payload is empty')
    groups, current, size = [], [], 22
    for record in records:
        if record[2] + 22 >= max_bytes:
            raise ValueError(f'One complete file exceeds the archive limit: {record[0]}')
        if current and size + record[2] >= max_bytes:
            groups.append(current)
            current, size = [], 22
        current.append(record)
        size += record[2]
    groups.append(current)
    destination.mkdir(parents=True)
    manifest = {'schema_version': 1, 'max_archive_bytes': ARCHIVE_LIMIT, 'archives': [], 'files': []}
    for index, group in enumerate(groups, 1):
        name = 'eval_stats.zip' if len(groups) == 1 else f'eval_stats__part{index:02d}.zip'
        target = destination / name
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, data, size in group:
                archive.writestr(_zip_info(relative), data)
                manifest['files'].append({'path': relative, 'archive': name, 'bytes': len(data),
                                          'sha256': hashlib.sha256(data).hexdigest()})
        payload = target.read_bytes()
        if len(payload) >= max_bytes:
            raise ValueError(f'Final archive exceeds configured limit: {target}')
        with zipfile.ZipFile(target) as archive:
            for relative, data, size in group:
                if archive.read(relative) != data or (source / relative).read_bytes() != data:
                    raise ValueError(f'Source/archive changed during packaging: {relative}')
        manifest['archives'].append({'path': name, 'bytes': len(payload),
                                     'sha256': hashlib.sha256(payload).hexdigest()})
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path, help='New staging directory')
    parser.add_argument('--max-bytes', type=int, default=ARCHIVE_LIMIT)
    args = parser.parse_args(argv)
    manifest = package(args.source, args.destination, args.max_bytes)
    print(f"Verified {len(manifest['files'])} files in {len(manifest['archives'])} ZIP archive(s).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
