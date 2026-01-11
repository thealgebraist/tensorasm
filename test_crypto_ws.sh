#!/bin/bash
# Integration test script for Crypto WebSocket Client

echo "================================================="
echo "  Crypto WebSocket Client - Integration Test"
echo "================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo -n "Testing: $test_name... "
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "Build Tests:"
echo "------------"

# Test 1: Clean build
run_test "Clean build directory" "make -f Makefile.crypto clean"

# Test 2: Compile main client
run_test "Compile crypto_ws_client.cpp" "make -f Makefile.crypto all"

# Test 3: Binary exists
run_test "Binary crypto_ws_client exists" "test -f crypto_ws_client && test -x crypto_ws_client"

# Test 4: Compile test client
run_test "Compile crypto_ws_test.cpp" "g++ -std=c++23 -o crypto_ws_test crypto_ws_test.cpp"

# Test 5: Test binary exists
run_test "Binary crypto_ws_test exists" "test -f crypto_ws_test && test -x crypto_ws_test"

echo ""
echo "Runtime Tests:"
echo "--------------"

# Test 6: Test program runs
run_test "Run crypto_ws_test successfully" "timeout 5 ./crypto_ws_test"

# Test 7: Main program handles no network gracefully
echo -n "Testing: Main program handles offline mode... "
if timeout 10 ./crypto_ws_client 2>&1 | grep -q "Testing Binance"; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "Code Quality Tests:"
echo "-------------------"

# Test 8: Check for C++23 features
echo -n "Testing: Uses std::format... "
if grep -q "std::format" crypto_ws_client.cpp; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

# Test 9: Check for chrono literals
echo -n "Testing: Uses chrono literals... "
if grep -q "using namespace std::chrono_literals" crypto_ws_client.cpp; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

# Test 10: Check for SSL/TLS support
echo -n "Testing: Includes OpenSSL headers... "
if grep -q "#include <openssl/ssl.h>" crypto_ws_client.cpp; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

# Test 11: Check for all 4 altcoins
echo -n "Testing: Monitors 4 altcoins (ETH, BNB, SOL, ADA)... "
if grep -q "ETHUSDT" crypto_ws_client.cpp && \
   grep -q "BNBUSDT" crypto_ws_client.cpp && \
   grep -q "SOLUSDT" crypto_ws_client.cpp && \
   grep -q "ADAUSDT" crypto_ws_client.cpp; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

# Test 12: Check for Binance support
echo -n "Testing: Supports Binance exchange... "
if grep -q "stream.binance.com" crypto_ws_client.cpp; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

# Test 13: Check for Bybit support
echo -n "Testing: Supports Bybit exchange... "
if grep -q "Bybit" crypto_ws_client.cpp; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((TESTS_FAILED++))
fi

echo ""
echo "Documentation Tests:"
echo "--------------------"

# Test 14: README exists
run_test "README file exists" "test -f CRYPTO_WS_README.md"

# Test 15: Makefile exists
run_test "Makefile exists" "test -f Makefile.crypto"

echo ""
echo "================================================="
echo "  Test Summary"
echo "================================================="
echo -e "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo -e "Total Tests:  $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed. ✗${NC}"
    exit 1
fi
