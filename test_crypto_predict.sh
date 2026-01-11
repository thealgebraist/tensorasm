#!/bin/bash
# Smoke test for crypto_predict.py

set -e

OUT_DIR="artifacts/predict_smoke"
ALTCOINS=("MATICUSDT" "AVAXUSDT" "LINKUSDT" "ATOMUSDT")

rm -rf "$OUT_DIR"

echo "Running altcoin forecaster (fast mode)..."
python3 crypto_predict.py --wait-seconds 0 --output-dir "$OUT_DIR" --altcoins "${ALTCOINS[@]}"

TEX_FILE="$OUT_DIR/altcoin_forecast.tex"
PDF_FILE="$OUT_DIR/altcoin_forecast.pdf"

if [ ! -f "$TEX_FILE" ]; then
    echo "Missing LaTeX output at $TEX_FILE"
    exit 1
fi

if [ ! -f "$PDF_FILE" ]; then
    echo "Missing PDF output at $PDF_FILE"
    exit 1
fi

for sym in "${ALTCOINS[@]}"; do
    if ! grep -q "$sym" "$TEX_FILE"; then
        echo "Symbol $sym not found in report"
        exit 1
    fi
done

echo "crypto_predict.py smoke test passed."
