#!/bin/bash
# reset.sh – Reset the invoice database and re-run the full pipeline.
# Usage:
#   ./reset.sh -clear                   Truncate the invoices table (deletes all rows)
#   ./reset.sh -process [path]          Run OCR pipeline on a directory or file
#                                       (default: ~/ocr/ if no path given)
#   ./reset.sh -embed                   Generate embeddings for all rows missing them
#   ./reset.sh -all                     Clear, process (default dir), and embed
#   ./reset.sh                          Show this help
#
# Examples:
#   ./reset.sh -process                          # process all images in ~/ocr/
#   ./reset.sh -process ~/my_invoices            # process a different directory
#   ./reset.sh -process ~/ocr/sample_invoice.jpg # process a single file

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Default directory for processing
DEFAULT_PROCESS_DIR="$HOME/ocr"

clear_db() {
    echo "🗑️  Truncating invoices table..."
    docker exec -it postgres psql -U ocr -d invoices -c "TRUNCATE TABLE invoices RESTART IDENTITY;"
    echo "✅ Table cleared."
}

process_path() {
    local target="$1"
    if [ -z "$target" ]; then
        target="$DEFAULT_PROCESS_DIR"
    fi

    echo "📄 Running OCR pipeline on: $target"
    if [ -f "$target" ]; then
        # Single file
        python3 pipeline_fast.py -f "$target"
    elif [ -d "$target" ]; then
        # Directory
        python3 pipeline_fast.py -d "$target"
    else
        echo "❌ Error: '$target' is not a valid file or directory."
        exit 1
    fi
    echo "✅ Processing finished."
}

embed_all() {
    echo "🧠 Generating embeddings for all unembedded rows..."
    python3 scripts/embed_update.py
    echo "✅ Embeddings generated."
}

show_help() {
    cat << EOF
Usage: $0 [option] [path]

Options:
  -clear          Truncate the invoices table (delete all rows)
  -process [path] Run OCR pipeline on a file or directory.
                  If no path is given, defaults to $DEFAULT_PROCESS_DIR
  -embed          Generate embeddings for rows missing them
  -all            Clear, process (default dir), and embed
  -h, --help      Show this help

Examples:
  $0 -process                          # process all images in $DEFAULT_PROCESS_DIR
  $0 -process ~/my_invoices            # process a different directory
  $0 -process invoice.jpg              # process a single file
  $0 -all                              # full reset with default directory
EOF
}

# Main
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    -clear)
        clear_db
        ;;
    -process)
        process_path "$2"
        ;;
    -embed)
        embed_all
        ;;
    -all)
        clear_db
        process_path "$2"   # optional directory for processing
        embed_all
        echo "🎉 Full reset complete."
        ;;
    -h|--help)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac