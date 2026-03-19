#!/bin/bash

set -euo pipefail

DEFAULT_URL="https://zenodo.org/records/19100654/files/data_root.zip?download=1"
DEFAULT_DEST_DIR="~/Downloads"


if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Usage: $0 [destination_dir] [zip_url]"
    echo
    echo "Example:"
    echo "  $0 /path/to/data_root"
    exit 0
fi


DEST_DIR="${1:-$DEFAULT_DEST_DIR}"
ZIP_URL="${2:-$DEFAULT_URL}"

mkdir -p "$DEST_DIR"
TMP_ZIP="$(mktemp /tmp/stoic_data_root_XXXXXX.zip)"

echo "Downloading dataset from:"
echo "  $ZIP_URL"

if command -v wget >/dev/null 2>&1; then
    wget -O "$TMP_ZIP" "$ZIP_URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L "$ZIP_URL" -o "$TMP_ZIP"
else
    echo "Error: neither wget nor curl is installed."
    rm -f "$TMP_ZIP"
    exit 1
fi

echo "Unzipping archive into:"
echo "  $DEST_DIR"
unzip -o "$TMP_ZIP" -d "$DEST_DIR"

rm -f "$TMP_ZIP"
echo "Done."
