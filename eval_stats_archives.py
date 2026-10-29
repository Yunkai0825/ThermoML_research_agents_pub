"""Restore published statistics, workflow, and figures; run once or use the browser."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile

from database_archives import (
    DatabaseArchiveError, _digest, _fingerprint, _integer,
    _preparation_lock, _relative_path, _sha256, _write_receipt,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent
ARCHIVE_LIMIT = 100_000_000


class EvalStatsArchiveError(RuntimeError):
    """Published statistics could not be restored without risking local files."""


def _safe_path(root, value):
    relative = _relative_path(value)
    reserved = {'con', 'prn', 'aux', 'nul'} | {f'{prefix}{n}' for prefix in ('com', 'lpt') for n in range(1, 10)}
    if any(part.rstrip(' .') != part or part.split('.')[0].casefold() in reserved for part in relative.parts):
        raise EvalStatsArchiveError(f'Unsafe filename: {value!r}')
    candidate = Path(root)
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink() or getattr(candidate, 'is_junction', lambda: False)():
            raise EvalStatsArchiveError(f'Symlink/junction is not a payload path: {candidate}')
    def canonical(path):
        value = str(path.resolve())
        # Windows adds an extended UNC prefix only for sufficiently deep paths.
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value)
    try:
        canonical(candidate).relative_to(canonical(Path(root)))
    except (OSError, ValueError) as exc:
        raise EvalStatsArchiveError(f'Path leaves statistics directory: {candidate}') from exc
    return candidate


def _manifest(directory):
    path = _safe_path(directory, 'manifest.json')
    if path.stat().st_size > 8 * 1024 * 1024:
        raise EvalStatsArchiveError('Statistics manifest is unexpectedly large')
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get('schema_version') != 1 or not isinstance(document.get('archives'), list) or not document['archives'] or not isinstance(document.get('files'), list) or not document['files']:
        raise EvalStatsArchiveError('Unsupported or empty statistics archive manifest')
    archives, files, folded = {}, {}, set()
    for row in document['archives']:
        name = row['path']
        if '/' in name or not name.endswith('.zip') or name.casefold() in folded:
            raise EvalStatsArchiveError(f'Invalid or duplicate archive name: {name!r}')
        _safe_path(directory, name)
        if _integer(row['bytes'], 'archive bytes', 1) >= ARCHIVE_LIMIT:
            raise EvalStatsArchiveError(f'Archive is not smaller than {ARCHIVE_LIMIT} bytes: {name}')
        _digest(row['sha256'], name)
        archives[name] = row
        folded.add(name.casefold())
    folded.clear()
    for row in document['files']:
        name = row['path']
        _safe_path(directory, name)
        parts = name.split('/')
        if not ((len(parts) == 1 and name in {'eval_stats.py', 'selection.json'})
                or (len(parts) > 1 and parts[0] in {'eval_workflow', 'numeric_outcomes'})):
            raise EvalStatsArchiveError(f'Unexpected statistics payload path: {name!r}')
        if name.casefold() in folded or row['archive'] not in archives:
            raise EvalStatsArchiveError(f'Duplicate member or unknown archive: {name!r}')
        _integer(row['bytes'], 'member bytes')
        _digest(row['sha256'], name)
        files[name] = row
        folded.add(name.casefold())
    for name in files:
        parts = name.split('/')
        if any('/'.join(parts[:i]).casefold() in folded for i in range(1, len(parts))):
            raise EvalStatsArchiveError(f'Conflicting file/directory paths: {name!r}')
    if {row['archive'] for row in files.values()} != set(archives):
        raise EvalStatsArchiveError('Archive without payload members')
    return archives, files, hashlib.sha256(payload).hexdigest()


def ensure_eval_stats(root=None, *, progress=None):
    """Verify/restore missing payload files; preserve existing edits and timestamps."""
    try:
        root = Path(root) if root is not None else REPOSITORY_ROOT
        directory = _safe_path(root, '_benchmark/_eval_stats')
        cache = _safe_path(root, '.eval_stats_cache')
        cache.mkdir(exist_ok=True)
        with _preparation_lock(cache, 120):
            archives, files, manifest_hash = _manifest(directory)
            archive_stamps = {name: _fingerprint(_safe_path(directory, name)) for name in archives}
            for name, row in archives.items():
                if archive_stamps[name]['bytes'] != row['bytes']:
                    raise EvalStatsArchiveError(f'Archive size mismatch: {name}')
            existing = {name: _fingerprint(_safe_path(directory, name)) for name in files
                        if _safe_path(directory, name).exists()}
            state = {'manifest_sha256': manifest_hash, 'archives': archive_stamps, 'files': existing}
            receipt = _safe_path(cache, 'prepared.json')
            if len(existing) == len(files) and receipt.is_file():
                try:
                    if json.loads(receipt.read_bytes()) == state:
                        return {'restored': 0, 'files': len(files)}
                except (ValueError, OSError):
                    pass
            # Preflight every existing destination before publishing any missing file.
            for name in existing:
                path = _safe_path(directory, name)
                if existing[name]['bytes'] != files[name]['bytes'] or _sha256(path) != files[name]['sha256']:
                    raise EvalStatsArchiveError(f'Local statistics file differs and was preserved: {path}. Move it aside before restoring published files.')
            restored = 0
            with tempfile.TemporaryDirectory(prefix='restore-', dir=cache) as temporary:
                staging = Path(temporary)
                for name, row in archives.items():
                    archive_path = _safe_path(directory, name)
                    if _sha256(archive_path) != row['sha256']:
                        raise EvalStatsArchiveError(f'Archive checksum mismatch: {name}')
                    expected = {key for key, member in files.items() if member['archive'] == name}
                    with zipfile.ZipFile(archive_path) as archive:
                        entries = archive.infolist()
                        if len(entries) != len(expected) or {info.orig_filename for info in entries} != expected:
                            raise EvalStatsArchiveError(f'Unexpected/duplicate ZIP members: {name}')
                        for info in entries:
                            member = info.orig_filename
                            target = _safe_path(directory, member)
                            row = files[member]
                            mode = stat.S_IFMT(info.external_attr >> 16)
                            if info.is_dir() or mode not in (0, stat.S_IFREG) or info.flag_bits & 1 or info.file_size != row['bytes']:
                                raise EvalStatsArchiveError(f'Invalid ZIP member: {member}')
                            staged = _safe_path(staging, member)
                            staged.parent.mkdir(parents=True, exist_ok=True)
                            digest, size = hashlib.sha256(), 0
                            with archive.open(info) as source, staged.open('wb') as output:
                                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                                    size += len(chunk)
                                    if size > row['bytes']:
                                        raise EvalStatsArchiveError(f'Oversized ZIP member: {member}')
                                    output.write(chunk)
                                    digest.update(chunk)
                            if size != row['bytes'] or digest.hexdigest() != row['sha256']:
                                raise EvalStatsArchiveError(f'Member checksum/size mismatch: {member}')
                # Only install after every archive and member has passed verification.
                for name in files.keys() - existing.keys():
                    target = _safe_path(directory, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target = _safe_path(directory, name)
                    staged = staging / name
                    if os.name == 'nt':
                        os.rename(staged, target)  # Refuses an existing destination on Windows.
                    else:
                        os.link(staged, target)    # Exclusive publication on other platforms.
                    restored += 1
            state['files'] = {name: _fingerprint(_safe_path(directory, name)) for name in files}
            if any(state['files'][name] != stamp for name, stamp in existing.items()):
                raise EvalStatsArchiveError('An existing statistics file changed during preparation; it was preserved.')
            _write_receipt(receipt, state)
            if progress and restored:
                progress(f'Prepared {restored} statistics/workflow files from the published ZIP payload.')
            return {'restored': restored, 'files': len(files)}
    except (DatabaseArchiveError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise EvalStatsArchiveError(f'Statistics preparation failed: {exc}') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPOSITORY_ROOT, help='Repository root (defaults to this checkout)')
    args = parser.parse_args(argv)
    try:
        result = ensure_eval_stats(args.root, progress=print)
    except EvalStatsArchiveError as exc:
        parser.exit(1, f'{exc}\n')
    print(f"Statistics payload ready: {result['files']} files.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
