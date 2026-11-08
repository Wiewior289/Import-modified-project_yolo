#!/usr/bin/env python3
"""Verify the NCNN YOLO11n graph boundary used by the async GPU/CPU split.

The parser mirrors blob-index allocation in NCNN Net::load_param(): unseen
bottoms are assigned indices first, then each top receives the next index.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

GPU_PREFIX_LAST_LAYER = 109
EXPECTED_LAYER_COUNT = 257
EXPECTED_BLOB_COUNT = 307

# blob_id: (producer index/name, first CPU-suffix consumer index/name)
EXPECTED_BOUNDARY = {
    37: (31, "splitncnn_2", 145, "cat_10"),
    72: (60, "splitncnn_6", 130, "cat_8"),
    123: (102, "split_4", 125, "cat_7"),
    125: (103, "splitncnn_13", 119, "add_8"),
    128: (105, "splitncnn_14", 114, "Tensor.view_6"),
    129: (105, "splitncnn_14", 115, "Tensor.reshape_4"),
    133: (107, "split_5", 110, "matmul_167"),
    135: (108, "splitncnn_15", 115, "Tensor.reshape_4"),
    136: (108, "splitncnn_15", 113, "matmultransb_0"),
    137: (109, "transpose_169", 110, "matmul_167"),
}


@dataclass(frozen=True)
class Layer:
    index: int
    layer_type: str
    name: str
    bottoms: tuple[int, ...]
    tops: tuple[int, ...]


def parse_param(path: Path) -> tuple[int, int, list[Layer]]:
    records = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(records) < 2:
        raise ValueError(".param is truncated: missing magic/count header")
    if records[0] != "7767517":
        raise ValueError(f"unexpected NCNN param magic: {records[0]!r}")
    layer_count, blob_count = map(int, records[1].split()[:2])

    blob_ids: dict[str, int] = {}
    next_blob_id = 0
    layers: list[Layer] = []

    for index, record in enumerate(records[2:]):
        tokens = record.split()
        if len(tokens) < 4:
            raise ValueError(f"malformed layer record at index {index}: {record}")
        bottom_count = int(tokens[2])
        top_count = int(tokens[3])
        begin_bottom = 4
        end_bottom = begin_bottom + bottom_count
        end_top = end_bottom + top_count
        if len(tokens) < end_top:
            raise ValueError(f"truncated blob list at layer {index}: {record}")

        bottoms: list[int] = []
        for blob_name in tokens[begin_bottom:end_bottom]:
            if blob_name not in blob_ids:
                blob_ids[blob_name] = next_blob_id
                next_blob_id += 1
            bottoms.append(blob_ids[blob_name])

        tops: list[int] = []
        for blob_name in tokens[end_bottom:end_top]:
            if blob_name in blob_ids:
                raise ValueError(f"blob top {blob_name!r} is redefined at layer {index}")
            blob_ids[blob_name] = next_blob_id
            tops.append(next_blob_id)
            next_blob_id += 1

        layers.append(Layer(index, tokens[0], tokens[1], tuple(bottoms), tuple(tops)))

    if len(layers) != layer_count:
        raise ValueError(f"header says {layer_count} layers, parsed {len(layers)}")
    if next_blob_id != blob_count:
        raise ValueError(
            f"header says {blob_count} blobs, NCNN load_param order resolves {next_blob_id}"
        )
    return layer_count, blob_count, layers


def compute_boundary(
    layers: list[Layer], cut: int
) -> dict[int, tuple[int, str, int, str]]:
    producers: dict[int, Layer] = {}
    consumers: dict[int, list[Layer]] = {}
    for layer in layers:
        for blob_id in layer.tops:
            if blob_id in producers:
                raise ValueError(f"blob {blob_id} has multiple producers")
            producers[blob_id] = layer
        for blob_id in layer.bottoms:
            consumers.setdefault(blob_id, []).append(layer)

    boundary: dict[int, tuple[int, str, int, str]] = {}
    for blob_id, producer in producers.items():
        if producer.index > cut:
            continue
        suffix_consumers = [c for c in consumers.get(blob_id, []) if c.index > cut]
        if suffix_consumers:
            first = min(suffix_consumers, key=lambda c: c.index)
            boundary[blob_id] = (
                producer.index, producer.name, first.index, first.name
            )
    return dict(sorted(boundary.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("param", type=Path, help="path to yolo11n.param")
    parser.add_argument(
        "--cut", type=int, default=GPU_PREFIX_LAST_LAYER,
        help=f"last GPU prefix layer (default: {GPU_PREFIX_LAST_LAYER})",
    )
    args = parser.parse_args()

    try:
        layer_count, blob_count, layers = parse_param(args.param)
        if (layer_count, blob_count) != (EXPECTED_LAYER_COUNT, EXPECTED_BLOB_COUNT):
            raise ValueError(
                f"unexpected graph dimensions {layer_count}/{blob_count}; "
                f"expected {EXPECTED_LAYER_COUNT}/{EXPECTED_BLOB_COUNT}"
            )
        actual = compute_boundary(layers, args.cut)
        if args.cut == GPU_PREFIX_LAST_LAYER and actual != EXPECTED_BOUNDARY:
            missing = sorted(set(EXPECTED_BOUNDARY) - set(actual))
            extra = sorted(set(actual) - set(EXPECTED_BOUNDARY))
            changed = {
                key: (EXPECTED_BOUNDARY[key], actual[key])
                for key in set(actual) & set(EXPECTED_BOUNDARY)
                if actual[key] != EXPECTED_BOUNDARY[key]
            }
            raise ValueError(
                f"boundary mismatch; missing={missing}, extra={extra}, changed={changed}"
            )

        print(
            f"PASS: {layer_count} layers, {blob_count} blobs; "
            f"GPU prefix 0..{args.cut}"
        )
        print(f"Boundary blobs: {len(actual)}")
        print("blob  producer                first CPU consumer")
        for blob_id, (pidx, pname, cidx, cname) in actual.items():
            print(f"{blob_id:>4}  {pidx:>3}:{pname:<20} {cidx:>3}:{cname}")
        return 0
    except (OSError, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
