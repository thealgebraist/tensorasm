# Cryptocurrency WebSocket Client (C++23)

A modern C++23 program that connects to Binance and Bybit WebSocket APIs to monitor real-time price data for top altcoins.

## Overview

This program demonstrates:
- **C++23 Standard**: Uses modern C++ features like `std::format`, chrono literals
- **WebSocket Protocol**: Full WebSocket client implementation with SSL/TLS
- **Multi-Exchange Support**: Connects to both Binance and Bybit
- **Real-time Data**: Streams live cryptocurrency prices

## Altcoins Monitored

The program tests 4 top altcoins:
1. **ETH** (Ethereum) - ETHUSDT
2. **BNB** (Binance Coin) - BNBUSDT
3. **SOL** (Solana) - SOLUSDT
4. **ADA** (Cardano) - ADAUSDT

## Features

### C++23 Features Used
- `std::format` for type-safe string formatting
- Chrono literals (`10s`, `500ms`, etc.)
- Improved lambda expressions
- Structured bindings for cleaner code

### Technical Implementation
- **SSL/TLS Encryption**: Secure WebSocket connections using OpenSSL
- **WebSocket Framing**: Proper frame parsing and handling
- **JSON Parsing**: Simple but effective ticker data extraction
- **Timeout Handling**: Configurable read timeouts
- **Multi-threading**: Thread-safe operations

## Building

### Prerequisites
- C++23 compatible compiler (GCC 13+, Clang 16+)
- OpenSSL development libraries
- Internet connection (for live data)

### Compilation

Using the provided Makefile:
```bash
make -f Makefile.crypto all
```

Or manually:
```bash
g++ -std=c++23 -Wall -Wextra -O2 -o crypto_ws_client crypto_ws_client.cpp -lssl -lcrypto
```

### Running

Execute the built binary:
```bash
./crypto_ws_client
```

For the test demonstration (works offline):
```bash
g++ -std=c++23 -o crypto_ws_test crypto_ws_test.cpp
./crypto_ws_test
```

## Architecture

### WebSocket Client Class
- **Connection Management**: Socket creation, SSL setup, handshake
- **Frame Reading**: WebSocket frame parsing with extended payload support
- **Error Handling**: Comprehensive error checking and reporting

### Exchange Integration

#### Binance
- **Endpoint**: `wss://stream.binance.com/ws/{symbol}@ticker`
- **Format**: Individual ticker streams for each trading pair
- **Data**: Real-time 24hr ticker statistics

#### Bybit
- **Endpoint**: `wss://stream.bybit.com/v5/public/spot`
- **Format**: Requires subscription messages after connection
- **Note**: Full implementation would need to send subscription JSON

## Code Structure

```
crypto_ws_client.cpp    - Main WebSocket client implementation
crypto_ws_test.cpp      - Offline test/demo program
Makefile.crypto         - Build configuration
```

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
[12:34:57] Binance @ ETH: $3251.20
...
```

## Protocol Details

### WebSocket Handshake
1. TCP connection to exchange
2. SSL/TLS negotiation
3. HTTP upgrade request with WebSocket headers
4. Server responds with 101 Switching Protocols

### Frame Format
- FIN bit: Final fragment
- Opcode: Message type (text, binary, close, etc.)
- Mask bit: Client-to-server masking
- Payload length: Variable length encoding (7-bit, 16-bit, or 64-bit)

## Security

- **TLS 1.2+**: All connections use modern TLS
- **Certificate Validation**: OpenSSL validates server certificates
- **Secure Protocols**: No plaintext WebSocket (ws://), only WSS (wss://)

## Limitations & Future Enhancements

### Current Limitations
- DNS resolution requires network access
- Bybit implementation is placeholder (needs subscription protocol)
- Simple JSON parsing (could use a JSON library)

### Potential Enhancements
- Add proper JSON parsing library (nlohmann/json, simdjson)
- Implement Bybit subscription messages
- Add order book depth monitoring
- Support more exchanges (Coinbase, Kraken, etc.)
- Add data persistence (database, files)
- Implement reconnection logic
- Add ping/pong keepalive messages

## License

This is example code for educational purposes.

## References

- [Binance WebSocket API](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
- [Bybit WebSocket API](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker)
- [WebSocket Protocol RFC 6455](https://datatracker.ietf.org/doc/html/rfc6455)
- [C++23 Standard](https://en.cppreference.com/w/cpp/23)
