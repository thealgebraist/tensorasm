#!/usr/bin/env python3
"""
Altcoin forecaster that fetches the last hour of prices for four medium-cap altcoins,
resamples them at 100ms resolution, predicts 10 seconds into the future, waits, and
compares against the realized price while exporting a LaTeX/PDF report.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

DEFAULT_ALTCOINS = ["MATICUSDT", "AVAXUSDT", "LINKUSDT", "ATOMUSDT"]
COINGECKO_IDS = {
    "MATICUSDT": "matic-network",
    "AVAXUSDT": "avalanche-2",
    "LINKUSDT": "chainlink",
    "ATOMUSDT": "cosmos",
}
CRYPTOCOMPARE_SYMBOLS = {
    "MATICUSDT": "MATIC",
    "AVAXUSDT": "AVAX",
    "LINKUSDT": "LINK",
    "ATOMUSDT": "ATOM",
}
STEP_MS = 100


def http_get_json(url: str, timeout: float = 5.0) -> Any:
    """Lightweight GET helper with short timeouts."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "tensorasm-crypto-forecaster/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_trades(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> List[Tuple[int, float]]:
    """
    Fetch aggregated trades from Binance. Falls back to an empty list on network errors.
    """
    base = "https://api.binance.com/api/v3/aggTrades"
    query = f"?symbol={symbol}&startTime={start_ms}&endTime={end_ms}&limit={limit}"
    try:
        data = http_get_json(base + query, timeout=5.0)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []

    trades: List[Tuple[int, float]] = []
    for item in data:
        try:
            trades.append((int(item["T"]), float(item["p"])))
        except (KeyError, ValueError, TypeError):
            continue
    return trades


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1m") -> List[Tuple[int, float]]:
    """
    Fetch candlesticks as a secondary source when trade data is unavailable.
    Uses the close price at the openTime of each kline.
    """
    base = "https://api.binance.com/api/v3/klines"
    query = f"?symbol={symbol}&interval={interval}&startTime={start_ms}&endTime={end_ms}&limit=1000"
    try:
        data = http_get_json(base + query, timeout=5.0)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []

    klines: List[Tuple[int, float]] = []
    for item in data:
        try:
            open_time = int(item[0])
            close_price = float(item[4])
            klines.append((open_time, close_price))
        except (ValueError, TypeError, IndexError):
            continue
    return klines


def fetch_coingecko_series(symbol: str) -> List[Tuple[int, float]]:
    """
    Fetch historical market data from CoinGecko as a fallback data source.
    Uses the last hour of minute-level prices.
    """
    coin_id = COINGECKO_IDS.get(symbol)
    if not coin_id:
        return []
    base = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    query = "?vs_currency=usd&days=1&interval=minute"
    try:
        data = http_get_json(base + query, timeout=5.0)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []

    prices = data.get("prices") or []
    if not prices:
        return []

    max_ts = max(int(entry[0]) for entry in prices if entry)
    window_start = max_ts - 3600 * 1000

    series: List[Tuple[int, float]] = []
    for entry in prices:
        try:
            ts = int(entry[0])
            price = float(entry[1])
            if ts >= window_start:
                series.append((ts, price))
        except (ValueError, TypeError, IndexError):
            continue
    return series


def fetch_cryptocompare_series(symbol: str) -> List[Tuple[int, float]]:
    """
    Fetch the last 60 minutes of minute-resolution prices from CryptoCompare.
    """
    base_symbol = CRYPTOCOMPARE_SYMBOLS.get(symbol, symbol.replace("USDT", ""))
    url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={base_symbol}&tsym=USD&limit=60"
    try:
        data = http_get_json(url, timeout=5.0)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return []

    if data.get("Response") != "Success":
        return []

    entries = data.get("Data", {}).get("Data", [])
    series: List[Tuple[int, float]] = []
    for item in entries:
        try:
            ts = int(item["time"]) * 1000
            price = float(item["close"])
            series.append((ts, price))
        except (KeyError, TypeError, ValueError):
            continue
    return series


def resample_100ms(
    trades: Sequence[Tuple[int, float]], start_ms: int, end_ms: int
) -> Tuple[List[Tuple[int, float]], bool]:
    """
    Resample trade ticks to 100ms cadence via forward-fill.
    Returns the resampled series and a flag indicating whether it is synthetic.
    """
    if not trades:
        raise RuntimeError("No trade data available for requested window.")

    ordered = sorted(trades, key=lambda x: x[0])
    idx = 0
    current_price = ordered[0][1]
    series: List[Tuple[int, float]] = []
    for ts in range(start_ms, end_ms + 1, STEP_MS):
        while idx < len(ordered) and ordered[idx][0] <= ts:
            current_price = ordered[idx][1]
            idx += 1
        series.append((ts, current_price))
    return series, False


def window_for_regression(series: Sequence[Tuple[int, float]], seconds: int = 30) -> List[Tuple[float, float]]:
    """Trim to the most recent window for regression, returning relative seconds."""
    if not series:
        return []
    window_points = max(1, int(seconds * 1000 / STEP_MS))
    tail = series[-window_points:]
    t0 = tail[0][0]
    return [((ts - t0) / 1000.0, price) for ts, price in tail]


def predict_linear(series: Sequence[Tuple[int, float]], horizon_seconds: float) -> float:
    """Simple linear regression prediction into the future."""
    data = window_for_regression(series)
    if not data:
        return 0.0
    times = [t for t, _ in data]
    prices = [p for _, p in data]
    mean_t = sum(times) / len(times)
    mean_p = sum(prices) / len(prices)
    denom = sum((t - mean_t) ** 2 for t in times)
    slope = (
        sum((t - mean_t) * (p - mean_p) for t, p in zip(times, prices)) / denom
        if denom
        else 0.0
    )
    intercept = mean_p - slope * mean_t
    future_t = times[-1] + horizon_seconds
    return max(0.0, intercept + slope * future_t)


def fetch_spot_price(symbol: str) -> float:
    """Fetch the latest spot price; returns NaN on failure."""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        data = http_get_json(url, timeout=3.0)
        return float(data.get("price"))
    except Exception:
        pass

    base_symbol = CRYPTOCOMPARE_SYMBOLS.get(symbol, symbol.replace("USDT", ""))
    cc_url = f"https://min-api.cryptocompare.com/data/price?fsym={base_symbol}&tsyms=USD"
    try:
        data = http_get_json(cc_url, timeout=3.0)
        price = data.get("USD")
        return float(price) if price is not None else float("nan")
    except Exception:
        pass

    coin_id = COINGECKO_IDS.get(symbol)
    if not coin_id:
        return float("nan")
    cg_url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        data = http_get_json(cg_url, timeout=3.0)
        price = data.get(coin_id, {}).get("usd")
        return float(price) if price is not None else float("nan")
    except Exception:
        return float("nan")


def render_table_rows(results: Iterable[dict]) -> str:
    rows = []
    for item in results:
        rows.append(
            f"{item['symbol']} & {item['last_price']:.4f} & "
            f"{item['predicted']:.4f} & {item['actual']:.4f} & "
            f"{item['error']:.4f} & {item['points']} & {item['source']}\\\\"
        )
    return "\n".join(rows)


def build_tex(results: Iterable[dict], horizon: float) -> str:
    table_rows = render_table_rows(results)
    return r"""\documentclass{article}
\usepackage{booktabs}
\usepackage[margin=1in]{geometry}
\begin{document}
\title{Altcoin 10s Forecast Report}
\author{tensorasm crypto toolkit}
\date{\today}
\maketitle
\section*{Method}
Prices were collected for the last hour at a 100\,ms cadence (forward-filled when live
ticks were sparse). A linear trend on the most recent window forecasts prices %0.1f seconds ahead.
After waiting the requested horizon the live spot price was sampled for comparison.
\section*{Results}
\begin{tabular}{lrrrrrl}
\toprule
Symbol & Last & Predicted & Actual & Error & Points & Source \\
\midrule
%s
\bottomrule
\end{tabular}
\end{document}
""" % (horizon, table_rows)


def write_tex(tex_body: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "altcoin_forecast.tex"
    tex_path.write_text(tex_body, encoding="utf-8")
    return tex_path


def compile_pdf(tex_path: Path, output_dir: Path) -> bool:
    """Compile LaTeX to PDF if pdflatex is available."""
    if not shutil.which("pdflatex"):
        return False
    cmd = [
        "pdflatex",
        "-interaction=batchmode",
        "-halt-on-error",
        "-output-directory",
        str(output_dir),
        str(tex_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def write_fallback_pdf(pdf_path: Path, lines: Sequence[str]) -> None:
    """Generate a minimal PDF without external dependencies."""
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = ["BT", "/F1 12 Tf", "72 760 Td"]
    for line in lines:
        content.append(f"({escape(line)}) Tj")
        content.append("0 -14 Td")
    content.append("ET")
    content_stream = "\n".join(content)
    content_bytes = content_stream.encode("utf-8")

    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content_bytes)} >>\nstream\n{content_stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    parts: List[str] = ["%PDF-1.4\n"]
    offsets: List[int] = []
    for idx, body in enumerate(objects, start=1):
        offsets.append(len("".join(parts).encode("utf-8")))
        parts.append(f"{idx} 0 obj\n{body}\nendobj\n")

    pre_xref = "".join(parts)
    xref_offset = len(pre_xref.encode("utf-8"))
    xref = ["xref\n0 6\n", "0000000000 65535 f \n"]
    for off in offsets:
        xref.append(f"{off:010d} 00000 n \n")
    xref.append("trailer << /Size 6 /Root 1 0 R >>\n")
    xref.append(f"startxref\n{xref_offset}\n%%EOF")

    pdf_bytes = (pre_xref + "".join(xref)).encode("utf-8")
    pdf_path.write_bytes(pdf_bytes)


def save_report(tex_body: str, output_dir: Path) -> Path:
    tex_path = write_tex(tex_body, output_dir)
    pdf_path = output_dir / "altcoin_forecast.pdf"
    if compile_pdf(tex_path, output_dir):
        return pdf_path
    summary = [
        "Altcoin 10s Forecast Report (fallback PDF)",
        f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"LaTeX source: {tex_path}",
    ]
    write_fallback_pdf(pdf_path, summary)
    return pdf_path


def process_symbol(symbol: str, wait_seconds: float, start_ms: int, end_ms: int) -> dict:
    trades = fetch_trades(symbol, start_ms, end_ms)
    source = "trades"
    if not trades:
        trades = fetch_klines(symbol, start_ms, end_ms)
        source = "klines"
    if not trades:
        trades = fetch_cryptocompare_series(symbol)
        source = "cryptocompare"
    if not trades:
        trades = fetch_coingecko_series(symbol)
        source = "coingecko"
    if not trades:
        raise RuntimeError(f"No market data available for {symbol}")
    max_ts = max(ts for ts, _ in trades)
    min_ts = min(ts for ts, _ in trades)
    start_ms_use = max(max_ts - 3600 * 1000, min_ts)
    end_ms_use = max_ts
    series, _ = resample_100ms(trades, start_ms_use, end_ms_use)
    predicted = predict_linear(series, wait_seconds or 10.0)
    last_price = series[-1][1] if series else float("nan")

    if wait_seconds > 0:
        time.sleep(wait_seconds)
    live_price = fetch_spot_price(symbol)
    if not live_price or math.isnan(live_price):
        raise RuntimeError(f"Failed to fetch live spot price for {symbol}")

    return {
        "symbol": symbol,
        "last_price": last_price,
        "predicted": predicted,
        "actual": live_price,
        "error": live_price - predicted,
        "points": len(series),
        "source": source,
    }


def run_forecast(altcoins: Sequence[str], wait_seconds: float, output_dir: Path) -> Path:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - 3600 * 1000
    results = []
    for symbol in altcoins:
        print(f"Collecting data for {symbol}...")
        results.append(process_symbol(symbol, wait_seconds, start_ms, end_ms))
    tex_body = build_tex(results, wait_seconds or 10.0)
    pdf_path = save_report(tex_body, output_dir)
    print(f"Report written to {pdf_path}")
    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Altcoin 10-second forecaster with LaTeX/PDF output.")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait before sampling the realized price (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/altcoin_forecast"),
        help="Directory for generated LaTeX/PDF assets.",
    )
    parser.add_argument(
        "--altcoins",
        type=str,
        nargs="*",
        default=DEFAULT_ALTCOINS,
        help="Override the default altcoin symbols.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_forecast(args.altcoins, args.wait_seconds, args.output_dir)


if __name__ == "__main__":
    main()
