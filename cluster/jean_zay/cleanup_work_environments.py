#!/usr/bin/env python3
"""Archive verified legacy WORK environments/caches in STORE, then remove them."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


TARGETS = (
    "venvs/vascular-graph-extraction",
    "venvs/vascular-graph-extraction-torch201-cu117",
    "venvs/vascular-graph-extraction-h100-torch231",
    "venvs/vascular-graph-extraction-a100-torch230",
    "venvs/vascular-graph-extraction-a100-torch230-fgw",
    ".cache/uv",
    ".cache/pip",
    ".cache/torch-extensions",
    ".cache/torch",
    ".cache/matplotlib",
    ".cache/fontconfig",
    ".cache/wandb",
    ".cache/huggingface",
)
OTHER_PROJECT_TARGETS = (
    "environments/greed-3d-vascular",
    "environments/vascular-graph-embedding-ged",
    "environments/vascular-graph-embedding-cpu",
)


def tree_entries(target: Path):
    """Visit entries without following directory symlinks."""
    yield target
    for directory, folders, files in os.walk(target, followlinks=False):
        for name in folders + files:
            yield Path(directory) / name


def discover_targets(work: Path, include_other_projects: bool) -> list[Path]:
    names = TARGETS + (OTHER_PROJECT_TARGETS if include_other_projects else ())
    targets = []
    for relative in names:
        target = work / relative
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink() or target.resolve() != target:
            print(f"Skipping symlink or redirected path: {target}")
        elif target.is_dir():
            targets.append(target)
    return targets


def check_idle() -> None:
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"],
        check=True, capture_output=True, text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Queued/running jobs exist; refusing cleanup until they finish:\n"
            + result.stdout
        )


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return digest_stream(handle)


def digest_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def assert_target(work: Path, target: Path) -> None:
    relative = target.relative_to(work).as_posix()
    if relative not in TARGETS + OTHER_PROJECT_TARGETS:
        raise RuntimeError(f"Path is not in the cleanup allowlist: {target}")
    if target.is_symlink() or target.resolve(strict=True) != target:
        raise RuntimeError(f"Source path was redirected: {target}")
    if not target.is_dir():
        raise RuntimeError(f"Source is not a directory: {target}")


def verify_archive(archive: Path, work: Path, target: Path) -> None:
    """Compare every member's bytes/type/link and reject missing/extra entries."""
    assert_target(work, target)
    relative = target.relative_to(work)
    expected = {path.relative_to(work).as_posix() for path in tree_entries(target)}
    seen = set()
    with tarfile.open(archive, "r:") as handle:
        for member in handle:
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            name.relative_to(relative)
            if member.name in seen:
                raise RuntimeError(f"Duplicate archive member: {member.name}")
            seen.add(member.name)
            source = work / name
            if member.issym():
                if not source.is_symlink() or os.readlink(source) != member.linkname:
                    raise RuntimeError(f"Symlink changed: {source}")
            elif member.isdir():
                if source.is_symlink() or not source.is_dir():
                    raise RuntimeError(f"Directory changed: {source}")
            elif member.isfile() or member.islnk():
                if source.is_symlink() or not source.is_file():
                    raise RuntimeError(f"File changed: {source}")
                if member.islnk():
                    link = Path(member.linkname)
                    if link.is_absolute() or ".." in link.parts:
                        raise RuntimeError(f"Unsafe hardlink: {member.linkname}")
                    link.relative_to(relative)
                archived_file = handle.extractfile(member)
                if archived_file is None:
                    raise RuntimeError(f"Cannot read archive member: {member.name}")
                with archived_file:
                    if digest_stream(archived_file) != digest_file(source):
                        raise RuntimeError(f"File contents changed: {source}")
            else:
                raise RuntimeError(f"Unsupported special file: {source}")
    if seen != expected:
        raise RuntimeError(
            f"Archive coverage mismatch for {target}: "
            f"missing={sorted(expected - seen)} extra={sorted(seen - expected)}"
        )


def archive_targets(work: Path, targets: list[Path], destination: Path) -> list[dict]:
    records = []
    for target in targets:
        assert_target(work, target)
        relative = target.relative_to(work).as_posix()
        archive = destination / (relative.replace("/", "__") + ".tar")
        partial = archive.with_suffix(".tar.partial")
        print(f"Archiving: {target}", flush=True)
        with tarfile.open(partial, "w", dereference=False) as handle:
            handle.add(target, arcname=relative)
        verify_archive(partial, work, target)
        partial.replace(archive)
        records.append({
            "source": str(target), "archive": str(archive),
            "sha256": digest_file(archive),
        })
    return records


def remove_archived_targets(work: Path, records: list[dict], idle_check=check_idle):
    for record in records:
        idle_check()
        target, archive = Path(record["source"]), Path(record["archive"])
        assert_target(work, target)
        if digest_file(archive) != record["sha256"]:
            raise RuntimeError(f"Archive checksum changed: {archive}")
        verify_archive(archive, work, target)
        shutil.rmtree(target)
        with (archive.parent / "removed.txt").open("a", encoding="utf-8") as handle:
            handle.write(str(target) + "\n")
        print(f"Removed (recoverable from {archive}): {target}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-other-projects", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("WORK"):
        parser.error("WORK must identify your existing Jean Zay work directory")
    work = Path(os.environ["WORK"]).resolve(strict=True)
    if not work.is_dir() or work in (Path("/"), Path.home().resolve()):
        raise RuntimeError(f"Refusing an unsafe WORK root: {work}")
    targets = discover_targets(work, args.include_other_projects)
    if not targets:
        print("No matching legacy WORK directories remain. Nothing to remove.")
        return 0
    print("Exact directories to archive and remove (individual inode counts):")
    for target in targets:
        inodes = {(path.lstat().st_dev, path.lstat().st_ino)
                  for path in tree_entries(target)}
        print(f"{len(inodes):9d}  {target}")
    print("No source trees, datasets, checkpoints, credentials or offline W&B run histories are targeted.")
    if not args.include_other_projects:
        print("GREED/embedding environments are excluded; use --include-other-projects to include the three known directories.")
    if not args.apply:
        print("Dry run only. No files were created or removed. Re-run with --apply to proceed.")
        return 0

    check_idle()
    active = os.environ.get("VIRTUAL_ENV")
    if active:
        active = Path(active).absolute()
        if any(active.is_relative_to(work / name) for name in ("venvs", "environments")) \
                or any(active.resolve().is_relative_to(work / name)
                       for name in ("venvs", "environments")):
            raise RuntimeError("A legacy WORK environment is active; deactivate or source the rebuilt SCRATCH environment before retrying")
    if not os.environ.get("STORE"):
        raise RuntimeError("STORE must identify your archive storage; no files removed")
    store = Path(os.environ["STORE"]).resolve(strict=True)
    if not store.is_dir() or store.is_relative_to(work) or store.stat().st_dev == work.stat().st_dev:
        raise RuntimeError("Archives must be outside WORK on a different filesystem")
    parent = store / "environment-archives"
    parent.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(tempfile.mkdtemp(prefix=f"work-cleanup-{timestamp}-", dir=parent))
    (destination / "README.txt").write_text(
        f"Original WORK: {work}\nRestore a directory with: tar -xf ARCHIVE.tar -C {work}\n"
        "Environments restored to their original path remain recoverable. Recreate environments when changing their installation path.\n",
        encoding="utf-8",
    )
    (destination / "targets.json").write_text(
        json.dumps([str(path) for path in targets], indent=2) + "\n", encoding="utf-8"
    )
    print(f"Recoverable archives: {destination}", flush=True)
    # Finish and verify ALL archives before deleting ANY source directory.
    records = archive_targets(work, targets, destination)
    (destination / "manifest.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    remove_archived_targets(work, records)
    print(f"Cleanup complete. Recovery records: {destination}")
    print("WORK's empty parent directories and small uv executable are retained.")
    print("Update other checkouts/project scripts before using them again.")
    print("Quota dashboards may take until their next refresh to reflect the reduction.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        raise SystemExit(f"Cleanup stopped: {type(error).__name__}: {error}")
