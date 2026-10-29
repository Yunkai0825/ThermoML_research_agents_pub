"""Read-only filesystem-like access to benchmark ZIPs, without extraction.

Only central-directory metadata is cached. Member payloads are streamed directly
from their archive, and every archive name is checked before it is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatchcase
from functools import lru_cache
import io
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat as stat_module
from types import MappingProxyType
from typing import Iterator, Mapping
import zipfile


def _member_name(value: str, *, allow_root: bool = True) -> str:
    """Accept one unambiguous relative POSIX path, never an OS escape."""
    value = str(value)
    if not value:
        if allow_root:
            return ""
        raise ValueError("Empty ZIP member name")
    if "\\" in value or "\x00" in value or value.startswith("/"):
        raise ValueError(f"Unsafe ZIP member path: {value!r}")
    if PureWindowsPath(value).drive:
        raise ValueError(f"Unsafe ZIP member drive: {value!r}")
    normalized = value[:-1] if value.endswith("/") else value
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"Unsafe ZIP member path: {value!r}")
    return normalized


@dataclass(frozen=True)
class _Index:
    entries: Mapping[str, zipfile.ZipInfo]
    directories: frozenset[str]
    children: Mapping[str, tuple[str, ...]]


@lru_cache(maxsize=32)
def _cached_index(archive: str, mtime_ns: int, size: int) -> _Index:
    # mtime and size deliberately form part of the cache key: replacement of an
    # archive invalidates its metadata without keeping any decompressed bytes.
    entries: dict[str, zipfile.ZipInfo] = {}
    directories = {""}
    with zipfile.ZipFile(archive, "r", allowZip64=True) as source:
        for info in source.infolist():
            name = _member_name(info.orig_filename, allow_root=False)
            if name in entries:
                raise ValueError(f"Duplicate ZIP member: {name!r}")
            if stat_module.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"ZIP symlinks are not supported: {name!r}")
            entries[name] = info
            if info.is_dir():
                directories.add(name)
            for parent in PurePosixPath(name).parents:
                directories.add("" if str(parent) == "." else parent.as_posix())
    for name in directories:
        if name in entries and not entries[name].is_dir():
            raise ValueError(f"ZIP member is both a file and directory: {name!r}")
    children: dict[str, list[str]] = {name: [] for name in directories}
    for name in set(entries) | directories:
        if name:
            parent = name.rpartition("/")[0]
            children[parent].append(name)
    return _Index(
        MappingProxyType(entries), frozenset(directories),
        MappingProxyType({name: tuple(sorted(items)) for name, items in children.items()}),
    )


def _physical_identity(value: Path) -> str:
    return os.path.normcase(str(value.resolve()))


def _matches(parts: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return _matches(parts, pattern[1:]) or bool(parts and _matches(parts[1:], pattern))
    return bool(parts and fnmatchcase(parts[0], pattern[0]) and _matches(parts[1:], pattern[1:]))


@dataclass(frozen=True)
class ZipPath:
    """A member path whose bytes remain inside a ZIP on disk.

    This intentionally does not implement ``__fspath__``: passing it to an OS
    filesystem API must fail instead of silently treating a member as a file.
    """

    archive: Path
    member: str = ""
    _archive_key: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "archive", Path(self.archive).resolve())
        object.__setattr__(self, "member", _member_name(self.member))
        object.__setattr__(self, "_archive_key", os.path.normcase(str(self.archive)))

    def _at(self, member: str) -> ZipPath:
        # Traversal and joins already validated these names. Reuse the resolved
        # archive identity instead of resolving the same UNC file per member.
        child = object.__new__(type(self))
        object.__setattr__(child, "archive", self.archive)
        object.__setattr__(child, "member", member)
        object.__setattr__(child, "_archive_key", self._archive_key)
        return child

    def _metadata(self) -> tuple[_Index, os.stat_result]:
        archive_stat = self.archive.stat()
        return _cached_index(self._archive_key, archive_stat.st_mtime_ns, archive_stat.st_size), archive_stat

    def _index(self) -> _Index:
        return self._metadata()[0]

    def __str__(self) -> str:
        return f"{self.archive}!/{self.member}"

    def as_posix(self) -> str:
        return f"{self.archive.as_posix()}!/{self.member}"

    def __truediv__(self, child: str) -> ZipPath:
        return self.joinpath(child)

    def joinpath(self, *children: str) -> ZipPath:
        parts = [self.member] if self.member else []
        for child in children:
            name = _member_name(str(child))
            if name:
                parts.append(name)
        return self._at("/".join(parts))

    @property
    def name(self) -> str:
        return self.member.rsplit("/", 1)[-1] if self.member else self.archive.stem

    @property
    def stem(self) -> str:
        return PurePosixPath(self.name).stem

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.name).suffix

    @property
    def parent(self) -> ZipPath | Path:
        if not self.member:
            return self.archive.parent
        return self._at(self.member.rpartition("/")[0])

    @property
    def parents(self) -> tuple[ZipPath | Path, ...]:
        result: list[ZipPath | Path] = []
        current = self.parent
        while isinstance(current, ZipPath):
            result.append(current)
            current = current.parent
        result.append(current)
        result.extend(current.parents)
        return tuple(result)

    def with_name(self, name: str) -> ZipPath:
        name = _member_name(name, allow_root=False)
        if "/" in name:
            raise ValueError(f"Invalid member basename: {name!r}")
        if not self.member:
            raise ValueError("Cannot rename the virtual ZIP root")
        return self.parent / name

    def with_suffix(self, suffix: str) -> ZipPath:
        return self.with_name(PurePosixPath(self.name).with_suffix(suffix).name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZipPath):
            return NotImplemented
        return self._archive_key == other._archive_key and self.member == other.member

    def __hash__(self) -> int:
        return hash((self._archive_key, self.member))

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (ZipPath, Path)):
            return path_identity(self) < path_identity(other)
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, (ZipPath, Path)):
            return path_identity(self) > path_identity(other)
        return NotImplemented

    def relative_to(self, base: ZipPath | Path | str) -> PurePosixPath:
        if isinstance(base, ZipPath):
            if self._archive_key != base._archive_key:
                raise ValueError(f"{self} is not in archive {base.archive}")
            return PurePosixPath(self.member).relative_to(PurePosixPath(base.member))
        physical_relative = self.archive.relative_to(Path(base).resolve())
        return PurePosixPath(physical_relative.as_posix()) / self.member

    def exists(self) -> bool:
        try:
            index = self._index()
        except FileNotFoundError:
            return False
        return self.member in index.entries or self.member in index.directories

    def is_file(self) -> bool:
        try:
            entry = self._index().entries.get(self.member)
        except FileNotFoundError:
            return False
        return entry is not None and not entry.is_dir()

    def is_dir(self) -> bool:
        try:
            return self.member in self._index().directories
        except FileNotFoundError:
            return False

    def is_symlink(self) -> bool:
        # Archive indexing rejects symlink members before exposing any paths.
        self._index()
        return False

    def iterdir(self) -> Iterator[ZipPath]:
        index = self._index()
        if self.member not in index.directories:
            if self.member in index.entries:
                raise NotADirectoryError(str(self))
            raise FileNotFoundError(str(self))
        for member in index.children[self.member]:
            yield self._at(member)

    def glob(self, pattern: str) -> Iterator[ZipPath]:
        pattern = str(pattern)
        only_directories = pattern.endswith("/")
        pattern = _member_name(pattern, allow_root=False)
        components = tuple(pattern.split("/"))
        if any("**" in part and part != "**" for part in components):
            raise ValueError("'**' must be an entire path component")
        index = self._index()
        if self.member not in index.directories:
            return
        # pathlib's terminal ** selects directories, including the current one.
        only_directories = only_directories or components[-1] == "**"
        pending = [self.member]
        while pending:
            member = pending.pop()
            is_directory = member in index.directories
            relative = member[len(self.member) + 1:] if self.member and member != self.member else member
            if member == self.member:
                relative = ""
            parts = tuple(relative.split("/")) if relative else ()
            if _matches(parts, components) and (is_directory or not only_directories):
                yield self._at(member)
            if is_directory:
                pending.extend(reversed(index.children[member]))

    def rglob(self, pattern: str) -> Iterator[ZipPath]:
        pattern = str(pattern)
        validated = _member_name(pattern, allow_root=False)
        yield from self.glob("**/" + validated + ("/" if pattern.endswith("/") else ""))

    def open(self, mode: str = "r", buffering: int = -1, encoding: str | None = None,
             errors: str | None = None, newline: str | None = None):
        if mode not in {"r", "rt", "rb"}:
            raise ValueError("ZIP benchmark paths are read-only")
        if mode == "rb" and any(value is not None for value in (encoding, errors, newline)):
            raise ValueError("Binary mode does not take encoding, errors, or newline")
        index = self._index()
        if self.member in index.directories:
            raise IsADirectoryError(str(self))
        info = index.entries.get(self.member)
        if info is None:
            raise FileNotFoundError(str(self))
        # ZipExtFile owns a reference to the archive handle, so the stream stays
        # valid after closing the ZipFile container and closes that handle itself.
        with zipfile.ZipFile(self.archive, "r", allowZip64=True) as source:
            stream = source.open(info, "r")
        if mode == "rb":
            return stream
        try:
            return io.TextIOWrapper(stream, encoding=encoding, errors=errors, newline=newline)
        except Exception:
            stream.close()
            raise

    def read_bytes(self) -> bytes:
        with self.open("rb") as source:
            return source.read()

    def read_text(self, encoding: str | None = None, errors: str | None = None,
                  newline: str | None = None) -> str:
        with self.open("r", encoding=encoding, errors=errors, newline=newline) as source:
            return source.read()

    def stat(self) -> os.stat_result:
        index, archive_stat = self._metadata()
        is_directory = self.member in index.directories
        info = index.entries.get(self.member)
        if not is_directory and info is None:
            raise FileNotFoundError(str(self))
        timestamp = archive_stat.st_mtime
        if info is not None:
            try:
                timestamp = datetime(*info.date_time).timestamp()
            except (ValueError, OSError, OverflowError):
                pass
        mode = (stat_module.S_IFDIR | 0o555) if is_directory else (stat_module.S_IFREG | 0o444)
        size = 0 if is_directory else info.file_size
        return os.stat_result((mode, 0, 0, 1, 0, 0, size, timestamp, timestamp, timestamp))


def as_path(value: ZipPath | Path | str) -> ZipPath | Path:
    return value if isinstance(value, ZipPath) else Path(value)


def path_identity(value: ZipPath | Path | str) -> str:
    if isinstance(value, ZipPath):
        return f"zip:{value._archive_key}!/{value.member}"
    return f"file:{_physical_identity(Path(value))}"


def is_within(candidate: ZipPath | Path | str, base: ZipPath | Path | str) -> bool:
    """Containment check respecting both physical roots and virtual members."""
    candidate, base = as_path(candidate), as_path(base)
    if isinstance(base, ZipPath):
        if not isinstance(candidate, ZipPath):
            return False
        try:
            candidate.relative_to(base)
        except ValueError:
            return False
        return True
    physical = candidate.archive if isinstance(candidate, ZipPath) else candidate
    try:
        physical.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True
