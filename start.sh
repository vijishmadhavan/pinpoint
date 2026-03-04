#!/bin/bash
# Pinpoint — Start both Python API and WhatsApp bot

echo "=== Starting Pinpoint ==="
echo ""

# Kill any existing instances (match exact script names)
pkill -f "python.*api\.py$" 2>/dev/null
pkill -f "node.*bot/index\.js$" 2>/dev/null
sleep 1

# Load nvm
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
if ! nvm use 20 2>/dev/null; then
    echo "[ERROR] Node.js 20 not found. Install with: nvm install 20"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start Python API in background
echo "[1/2] Starting Python API on port 5123..."
conda run -n pinpoint python "$SCRIPT_DIR/api.py" &
API_PID=$!

# Cleanup on exit (SIGINT, SIGTERM, normal exit)
trap "kill $API_PID 2>/dev/null; echo 'Pinpoint stopped.'" EXIT

# Wait for API to be ready
for i in $(seq 1 10); do
    if curl -s http://localhost:5123/ping > /dev/null 2>&1; then
        echo "[1/2] Python API ready!"
        break
    fi
    if [ "$i" -eq 10 ]; then
        echo "[ERROR] Python API failed to start after 10s. Check conda env 'pinpoint'."
        exit 1
    fi
    sleep 1
done

# Start WhatsApp bot (foreground)
echo "[2/2] Starting WhatsApp bot..."
echo ""
cd "$SCRIPT_DIR/bot" && node index.js
