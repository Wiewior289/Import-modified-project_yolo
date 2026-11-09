#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA="6a1bf000f363714839a36793addc8c879d3d899e"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/../patches/ncnn/0001-ahb-normalized-ycbcr-sampling.patch"
ROOT="${1:-$PWD}"
ROOT="$(cd "$ROOT" && pwd)"

if [[ ! -f "$ROOT/src/pipeline.cpp" || ! -f "$ROOT/src/allocator.cpp" || ! -f "$ROOT/src/convert_ycbcr.comp" ]]; then
    echo "ERROR: '$ROOT' nie wygląda na źródła NCNN z pipeline.cpp, allocator.cpp i convert_ycbcr.comp." >&2
    exit 2
fi

ACTUAL_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    echo "ERROR: oczekiwany NCNN commit $EXPECTED_SHA, otrzymano ${ACTUAL_SHA:-<brak git>}." >&2
    echo "Patch nie został zastosowany." >&2
    exit 3
fi

if git -C "$ROOT" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
    echo "OK: patch AHB jest już zastosowany. Bez zmian."
    exit 0
fi

git -C "$ROOT" apply --check "$PATCH_FILE"
git -C "$ROOT" apply "$PATCH_FILE"
echo "OK: zastosowano poprawki AHB normalized YCbCr sampling."
echo "Sprawdź: git -C '$ROOT' diff --check && git -C '$ROOT' diff --stat"
