#!/bin/bash
# Start script for PicDetect
# This starts both the API proxy and web server

echo "🚀 Starting PicDetect..."
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install flask flask-cors pillow transformers torch --quiet
fi

# Start API proxy in background
echo "🔧 Starting API proxy server (port 8001)..."
python3 api_proxy.py &
PROXY_PID=$!

# Wait a moment for proxy to start
sleep 2

# Start web server in background
echo "🌐 Starting web server (port 8000)..."
python3 server.py &
WEB_PID=$!

# Wait a moment
sleep 2

echo ""
echo "✅ Both servers are running!"
echo "📝 Open http://localhost:8000 in your browser"
echo ""
echo "⏹️  Press Ctrl+C to stop both servers"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $PROXY_PID $WEB_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait


