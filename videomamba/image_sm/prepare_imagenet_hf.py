#!/usr/bin/env python3
"""Convert Hugging Face ImageNet-1K parquet shards for this repository."""

import argparse
import concurrent.futures
import os
from pathlib import Path
import re
import tempfile

import pyarrow.parquet as pq


SHARD_PATTERN = re.compile(r"^(train|validation)-(\d+)-of-(\d+)\.parquet$")
EXPECTED_EXAMPLES = {"train": 1_281_167, "validation": 50_000}


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Extract ILSVRC/imagenet-1k parquet shards into images and meta files."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory created by `hf download ILSVRC/imagenet-1k --local-dir ...`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory containing train/, val/, and meta/.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Allow a nonstandard subset instead of requiring official train/val counts.",
    )
    return parser


def discover_shards(source, split):
    shards = []
    for path in (source / "data").glob(f"{split}-*.parquet"):
        match = SHARD_PATTERN.match(path.name)
        if match is not None:
            shards.append((int(match.group(2)), int(match.group(3)), path))

    if not shards:
        raise FileNotFoundError(f"No {split} parquet shards found under {source / 'data'}")

    shards.sort()
    totals = {total for _, total, _ in shards}
    if len(totals) != 1:
        raise RuntimeError(f"Inconsistent shard totals for {split}: {sorted(totals)}")
    total = totals.pop()
    indices = [index for index, _, _ in shards]
    if indices != list(range(total)):
        missing = sorted(set(range(total)) - set(indices))
        raise RuntimeError(
            f"Incomplete {split} download: found {len(shards)}/{total} shards; "
            f"first missing indices: {missing[:10]}"
        )
    return shards


def atomic_write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def extract_shard(task):
    split, shard_index, shard_path, output = task
    output_split = "val" if split == "validation" else split
    fragment_dir = output / "meta" / "fragments" / output_split
    fragment_path = fragment_dir / f"{shard_index:05d}.txt"
    marker_path = fragment_dir / f"{shard_index:05d}.complete"

    if marker_path.is_file() and fragment_path.is_file():
        with fragment_path.open("r", encoding="utf-8") as stream:
            return split, shard_index, sum(1 for _ in stream), True

    parquet = pq.ParquetFile(shard_path)
    meta_lines = []
    row_index = 0
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["image", "label"])
        images = table.column("image").combine_chunks()
        image_bytes = images.field("bytes")
        image_paths = images.field("path")
        labels = table.column("label").combine_chunks()

        for index in range(len(table)):
            data = image_bytes[index].as_py()
            original_name = image_paths[index].as_py() or "image.JPEG"
            label = labels[index].as_py()
            if not data:
                raise RuntimeError(f"Empty image bytes in {shard_path}, row {row_index}")
            if label is None or not 0 <= label < 1000:
                raise RuntimeError(f"Invalid label {label} in {shard_path}, row {row_index}")

            suffix = Path(original_name).suffix or ".JPEG"
            relative_path = Path(f"{label:04d}") / (
                f"{output_split}-{shard_index:05d}-{row_index:06d}{suffix}"
            )
            image_path = output / output_split / relative_path
            if not image_path.is_file() or image_path.stat().st_size != len(data):
                atomic_write_bytes(image_path, data)
            meta_lines.append(f"{relative_path.as_posix()} {label}\n")
            row_index += 1

    fragment_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(fragment_path, "".join(meta_lines))
    atomic_write_text(marker_path, f"{row_index}\n")
    return split, shard_index, row_index, False


def assemble_meta(output, split, shard_count):
    output_split = "val" if split == "validation" else split
    fragment_dir = output / "meta" / "fragments" / output_split
    final_path = output / "meta" / f"{output_split}.txt"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{final_path.name}.", dir=final_path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_stream:
            for shard_index in range(shard_count):
                fragment_path = fragment_dir / f"{shard_index:05d}.txt"
                if not fragment_path.is_file():
                    raise RuntimeError(f"Missing meta fragment: {fragment_path}")
                with fragment_path.open("r", encoding="utf-8") as input_stream:
                    for line in input_stream:
                        output_stream.write(line)
                        count += 1
        os.replace(temporary_name, final_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return final_path, count


def main():
    args = get_args_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    args.source = args.source.resolve()
    args.output = args.output.resolve()
    all_shards = {}
    tasks = []
    for split in ("train", "validation"):
        shards = discover_shards(args.source, split)
        all_shards[split] = shards
        tasks.extend(
            (split, shard_index, shard_path, args.output)
            for shard_index, _, shard_path in shards
        )
        print(f"Found all {len(shards)} {split} shards")

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_shard, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            split, shard_index, count, skipped = future.result()
            completed += 1
            state = "verified" if skipped else "extracted"
            print(
                f"[{completed}/{len(tasks)}] {state} {split} shard "
                f"{shard_index:05d}: {count} images",
                flush=True,
            )

    for split, shards in all_shards.items():
        meta_path, count = assemble_meta(args.output, split, len(shards))
        expected = EXPECTED_EXAMPLES[split]
        if not args.skip_count_check and count != expected:
            raise RuntimeError(f"Expected {expected} {split} examples, found {count}")
        print(f"Wrote {count} entries to {meta_path}")


if __name__ == "__main__":
    main()
