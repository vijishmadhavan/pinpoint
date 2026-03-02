#!/bin/bash
# Pinpoint — Start both Python API and WhatsApp bot

echo "=== Starting Pinpoint ==="
echo ""

# Kill any existing instances
pkill -f "python.*api.py" 2>/dev/null
pkill -f "node.*bot/index.js" 2>/dev/null
sleep 1

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm use 20 2>/dev/null

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start Python API in background
echo "[1/2] Starting Python API on port 5123..."
conda run -n pinpoint python "$SCRIPT_DIR/api.py" &
API_PID=$!

# Wait for API to be ready
for i in $(seq 1 10); do
    if curl -s http://localhost:5123/ping > /dev/null 2>&1; then
        echo "[1/2] Python API ready!"
        break
    fi
    sleep 1
done

# Start WhatsApp bot (foreground)
echo "[2/2] Starting WhatsApp bot..."
echo ""
cd "$SCRIPT_DIR/bot" && node index.js

# Cleanup on exit
kill $API_PID 2>/dev/null
echo "Pinpoint stopped."
