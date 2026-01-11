# Quick Start Guide - Crypto WebSocket Client

## Build & Run

### Option 1: Quick Test (Offline)
```bash
# Compile and run the test program
g++ -std=c++23 -o crypto_ws_test crypto_ws_test.cpp
./crypto_ws_test
```

### Option 2: Full Client (Requires Internet)
```bash
# Build using Makefile
make -f Makefile.crypto all

# Run the WebSocket client
./crypto_ws_client
```

### Option 3: Run All Tests
```bash
# Run comprehensive integration tests
chmod +x test_crypto_ws.sh
./test_crypto_ws.sh
```

## What It Does

Connects to Binance and Bybit cryptocurrency exchanges via WebSocket to monitor real-time prices for:
- **ETH** (Ethereum)
- **BNB** (Binance Coin)
- **SOL** (Solana)
- **ADA** (Cardano)

## Requirements

- C++23 compiler (GCC 13+ or Clang 16+)
- OpenSSL development libraries
- Internet connection (for live data)

## File Overview

| File | Purpose |
|------|---------|
| `crypto_ws_client.cpp` | Main WebSocket client implementation |
| `crypto_ws_test.cpp` | Test program (works offline) |
| `Makefile.crypto` | Build configuration |
| `CRYPTO_WS_README.md` | Full documentation |
| `test_crypto_ws.sh` | Integration test suite |

## Example Output

```
=================================================
  C++23 Crypto WebSocket Monitor
  Testing Binance & Bybit with 4 Top Altcoins
=================================================

--- Testing Binance Exchange ---
Connecting to Binance for ETH...
✓ Connected to stream.binance.com:443/ws/ethusdt@ticker
[12:34:56] Binance @ ETH: $3250.45
...
```

## Troubleshooting

**No internet connection?**
- Run `crypto_ws_test` instead for offline demonstration

**Compilation errors?**
- Ensure GCC 13+ or Clang 16+ is installed
- Check that OpenSSL development libraries are available:
  ```bash
  sudo apt-get install libssl-dev  # Ubuntu/Debian
  brew install openssl              # macOS
  ```

**Cannot resolve hostname?**
- Check internet connectivity
- Verify DNS is working: `ping stream.binance.com`

## More Information

See `CRYPTO_WS_README.md` for complete documentation.

## Altcoin Forecast Pipeline (PDF)

Run the Python forecaster to generate a live altcoin report with LaTeX/PDF output:

```bash
python3 crypto_predict.py --wait-seconds 0 --output-dir artifacts/altcoin_forecast
```

The pipeline fetches prices for MATIC, AVAX, LINK, and ATOM, writes `altcoin_forecast.tex`, and compiles `altcoin_forecast.pdf` (with a fallback PDF if LaTeX is unavailable). Use `--loop` to repeat the forecast on an interval.
