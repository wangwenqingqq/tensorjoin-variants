#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${1:-@TENSORJOIN_ROOT@/tensorjoin_20260902_certified_exact_join}
COMMIT=33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6
ARCHIVE="$PROJECT_ROOT/raw/ESC-50-$COMMIT.tar.gz"
EXTRACTED="$PROJECT_ROOT/raw/ESC-50-$COMMIT"
RECEIPT="$PROJECT_ROOT/receipts/esc50_source.txt"
URL="https://codeload.github.com/karolpiczak/ESC-50/tar.gz/$COMMIT"

mkdir -p "$PROJECT_ROOT/raw" "$PROJECT_ROOT/receipts"
if [[ ! -f "$ARCHIVE" ]]; then
  curl --fail --location --retry 5 --retry-delay 3 --continue-at - \
    --output "$ARCHIVE" "$URL"
fi

SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
if [[ ! -d "$EXTRACTED/audio" ]]; then
  TMP="$PROJECT_ROOT/raw/.extract-$COMMIT"
  rm -rf "$TMP"
  mkdir -p "$TMP"
  tar -xzf "$ARCHIVE" -C "$TMP"
  INNER=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  mv "$INNER" "$EXTRACTED"
  rmdir "$TMP"
fi

WAV_COUNT=$(find "$EXTRACTED/audio" -type f -name '*.wav' | wc -l | tr -d ' ')
{
  echo "url=$URL"
  echo "commit=$COMMIT"
  echo "archive=$ARCHIVE"
  echo "sha256=$SHA256"
  echo "extracted=$EXTRACTED"
  echo "wav_count=$WAV_COUNT"
  echo "retrieved_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RECEIPT"

if [[ "$WAV_COUNT" != "2000" ]]; then
  echo "Expected 2000 WAV files, found $WAV_COUNT" >&2
  exit 1
fi
cat "$RECEIPT"
