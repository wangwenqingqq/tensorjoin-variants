#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join}
URL=https://hf-mirror.com/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip
UPSTREAM_PAGE=https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/tree/04d4e5ca1dc93606cb58752b0c08331e598743a4
ARCHIVE="$ROOT/raw/UCF-101.zip"
OUT="$ROOT/data/ucf101"
RECEIPT="$ROOT/receipts/ucf101_source.txt"
EXPECTED_BYTES=6957373664
EXPECTED_SHA256=eb77e54dfafd9c77e7b086f5e5ef7738cabd16a701b3f5c301a1f6c81fc4756e

mkdir -p "$ROOT/raw" "$ROOT/data" "$ROOT/receipts" "$OUT"
{
  echo "url=$URL"
  echo "upstream_page=$UPSTREAM_PAGE"
  echo "upstream_commit=04d4e5ca1dc93606cb58752b0c08331e598743a4"
  echo "retrieved_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "expected_bytes=$EXPECTED_BYTES"
  echo "expected_sha256=$EXPECTED_SHA256"
  echo "provenance_note=community ZIP reports an unmodified conversion of official UCF101; artifact identity is frozen by the Hugging Face linked ETag"
} > "$RECEIPT"

# The fixed Hugging Face artifact is reached through hf-mirror.com because the
# direct endpoint resets connections from this host. The resolver supplies the
# pinned repository commit, linked size, and SHA-256-linked ETag.
curl -L --fail --retry 10 --retry-all-errors --continue-at - \
  --dump-header "$ROOT/receipts/ucf101_http_headers.txt" \
  --output "$ARCHIVE" "$URL"

actual_bytes=$(stat -c %s "$ARCHIVE")
if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
  echo "Unexpected archive size: $actual_bytes" >&2
  exit 2
fi
actual_sha256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
echo "$actual_sha256  $ARCHIVE" | tee -a "$RECEIPT"
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
  echo "Unexpected archive SHA-256: $actual_sha256" >&2
  exit 4
fi
echo "actual_bytes=$actual_bytes" >> "$RECEIPT"

7z x -y -o"$OUT" "$ARCHIVE"
video_count=$(find "$OUT" -type f -name '*.avi' | wc -l)
echo "video_count=$video_count" | tee -a "$RECEIPT"
if [[ "$video_count" != "13320" ]]; then
  echo "Expected 13320 AVI files, found $video_count" >&2
  exit 3
fi
find "$OUT" -type f -name '*.avi' -printf '%P\n' | LC_ALL=C sort \
  > "$ROOT/receipts/ucf101_videos_sorted.txt"
sha256sum "$ROOT/receipts/ucf101_videos_sorted.txt" | tee -a "$RECEIPT"
echo "ACQUIRE_COMPLETE archive=$ARCHIVE output=$OUT videos=$video_count"
