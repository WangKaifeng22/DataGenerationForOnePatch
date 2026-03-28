import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


SAMPLE_RE = re.compile(r"^sample_(\d+)\.(npy|npz)$")


@dataclass
class SampleFile:
    path: str
    index: int
    ext: str


def _scan_source(source_dir: str) -> List[SampleFile]:
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source not found: {source_dir}")

    items: List[SampleFile] = []
    for name in os.listdir(source_dir):
        match = SAMPLE_RE.match(name)
        if not match:
            continue
        index = int(match.group(1))
        ext = match.group(2)
        items.append(SampleFile(path=os.path.join(source_dir, name), index=index, ext=ext))

    if not items:
        raise ValueError(f"No sample files found in: {source_dir}")

    items.sort(key=lambda x: x.index)
    return items


def _check_continuity(items: List[SampleFile]) -> Tuple[bool, List[int]]:
    missing: List[int] = []
    if not items:
        return True, missing
    start = items[0].index
    end = items[-1].index
    seen = {it.index for it in items}
    for idx in range(start, end + 1):
        if idx not in seen:
            missing.append(idx)
    return len(missing) == 0, missing


def _infer_kind(exts: List[str]) -> str:
    unique = sorted(set(exts))
    if len(unique) != 1:
        raise ValueError(f"Mixed extensions found: {unique}")
    if unique[0] == "npy":
        return "sos"
    if unique[0] == "npz":
        return "kwave"
    raise ValueError(f"Unsupported extension: {unique[0]}")


def _validate_sos(path: str) -> None:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Invalid npy content: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"Expected float dtype in {path}, got {arr.dtype}")


def _validate_kwave(path: str) -> None:
    data = np.load(path)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError(f"Invalid npz content: {path}")
    required = {"time_data_cat", "sensor_coords"}
    if not required.issubset(set(data.files)):
        raise ValueError(f"Missing keys in {path}: {required - set(data.files)}")
    time_data = data["time_data_cat"]
    sensor_coords = data["sensor_coords"]
    if time_data.ndim != 3:
        raise ValueError(f"Expected 3D time_data_cat in {path}, got {time_data.shape}")
    if sensor_coords.ndim != 2:
        raise ValueError(f"Expected 2D sensor_coords in {path}, got {sensor_coords.shape}")


def _validate_file(path: str, kind: str) -> None:
    if kind == "sos":
        _validate_sos(path)
    elif kind == "kwave":
        _validate_kwave(path)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def _plan_merge(
    sources: List[str],
    allow_gaps: bool,
    kind: str,
) -> Tuple[List[SampleFile], str]:
    all_items: List[SampleFile] = []
    all_exts: List[str] = []

    for src in sources:
        items = _scan_source(src)
        ok, missing = _check_continuity(items)
        if not ok and not allow_gaps:
            raise ValueError(f"Missing indices in {src}: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        all_items.extend(items)
        all_exts.extend([it.ext for it in items])

    inferred_kind = _infer_kind(all_exts)
    if kind != "auto" and kind != inferred_kind:
        raise ValueError(f"Kind mismatch: expected {kind}, inferred {inferred_kind}")

    return all_items, inferred_kind


def merge_datasets(
    sources: List[str],
    output_dir: str,
    start_index: int,
    pad_width: int,
    allow_gaps: bool,
    kind: str,
    validate: bool,
    overwrite: bool,
    move: bool,
    dry_run: bool,
) -> None:
    items, inferred_kind = _plan_merge(sources, allow_gaps, kind)

    os.makedirs(output_dir, exist_ok=True)

    total = len(items)
    if total == 0:
        raise ValueError("No files to merge")

    action = "move" if move else "copy"
    print(f"Merging {total} files as {inferred_kind} ({action})")

    current_index = start_index
    for it in items:
        ext = it.ext
        dst_name = f"sample_{current_index:0{pad_width}d}.{ext}"
        dst_path = os.path.join(output_dir, dst_name)

        if os.path.exists(dst_path) and not overwrite:
            raise FileExistsError(f"Output exists: {dst_path}")

        if validate:
            _validate_file(it.path, inferred_kind)

        if dry_run:
            print(f"DRY-RUN: {it.path} -> {dst_path}")
        else:
            if move:
                shutil.move(it.path, dst_path)
            else:
                shutil.copy2(it.path, dst_path)

        current_index += 1

    print(f"Done. Wrote {total} files to: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sequential sample_* files from multiple sources into one folder.")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source folders, in the order they should be merged.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder to write merged samples.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index for the merged sequence.")
    parser.add_argument(
        "--pad-width",
        type=int,
        default=6,
        help="Zero padding width for output filenames.")
    parser.add_argument(
        "--kind",
        choices=["auto", "sos", "kwave"],
        default="auto",
        help="Dataset kind. Auto infers from extension.")
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Allow missing indices inside a source folder.")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip content validation for faster merges.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in output if they already exist.")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying/moving files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    merge_datasets(
        sources=args.sources,
        output_dir=args.output,
        start_index=args.start_index,
        pad_width=args.pad_width,
        allow_gaps=args.allow_gaps,
        kind=args.kind,
        validate=not args.no_validate,
        overwrite=args.overwrite,
        move=args.move,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

