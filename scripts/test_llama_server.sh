#!/bin/bash
# test_server_ocr.sh – test llama-server with your exact config
# Usage:
#   ./test_server_ocr.sh <image.jpg>               # auto-encodes the image
#   ./test_server_ocr.sh -b <base64_file>          # uses existing base64 file (like test_b64.txt)
#
# Example:
#   ./test_server_ocr.sh sample_invoice.jpg
#   ./test_server_ocr.sh -b test_b64.txt

set -e

# ── Server settings (unchanged) ──
URL="http://127.0.0.1:8081/v1/chat/completions"
MODEL="Unlimited-OCR"
PROMPT="Please OCR the text in this image."
TEMPERATURE=0.1
MAX_TOKENS=512
REPEAT_PENALTY=1.1
STREAM="false"        # JSON boolean

# ── Image handling ──
if [ "$1" = "-b" ]; then
    # Read base64 from a file (remove newlines)
    if [ -z "$2" ]; then
        echo "Error: -b requires a base64 file path"
        exit 1
    fi
    B64=$(tr -d '\n' < "$2")
    echo "Using base64 file: $2 (${#B64} chars)"
else
    IMAGE="$1"
    if [ -z "$IMAGE" ]; then
        echo "Usage: $0 <image.jpg>   OR   $0 -b <base64_file>"
        exit 1
    fi
    B64=$(base64 -w0 "$IMAGE")
    echo "Encoding image: $IMAGE (${#B64} chars)"
fi

# ── Send request ──
echo ""
echo "Sending request to $URL ..."
RESPONSE=$(curl -s -N "$URL" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [
            {
                \"role\": \"user\",
                \"content\": [
                    {
                        \"type\": \"text\",
                        \"text\": \"$PROMPT\"
                    },
                    {
                        \"type\": \"image_url\",
                        \"image_url\": {
                            \"url\": \"data:image/jpeg;base64,$B64\"
                        }
                    }
                ]
            }
        ],
        \"model\": \"$MODEL\",
        \"temperature\": $TEMPERATURE,
        \"max_tokens\": $MAX_TOKENS,
        \"repeat_penalty\": $REPEAT_PENALTY,
        \"stream\": $STREAM
    }")

# ── Display result ──
echo "Server response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

# Optionally extract the content field
echo ""
echo "=== Extracted content ==="
echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['message']['content'])
except Exception as e:
    print('(could not extract content: {})'.format(e))
"