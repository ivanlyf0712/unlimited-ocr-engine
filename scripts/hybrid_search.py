#!/usr/bin/env python3
"""
Hybrid semantic + keyword search over invoices – command‑line interface.
Usage:
  python3 hybrid_search.py "your question" [vendor_filter] [limit]
  python3 hybrid_search.py -q "total amount" -v Alibaba -k "shipping" -d 2024-01-01 -D 2024-12-31
Options:
  -q, --query       Natural language query (required)
  -v, --vendor      Vendor name ILIKE filter
  -k, --keyword     Full‑text search on raw_text (PostgreSQL tsquery)
  -d, --date-from   Start date filter (YYYY-MM-DD)
  -D, --date-to     End date filter (YYYY-MM-DD)
  -a, --amount-min  Minimum total_amount
  -A, --amount-max  Maximum total_amount
  -l, --limit       Number of results (default: 5)
"""

import argparse
from core.db import search_similar, get_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid semantic search over invoices")
    parser.add_argument("-q", "--query", required=True, help="Natural language query")
    parser.add_argument("-v", "--vendor", default=None, help="Vendor name filter (ILIKE)")
    parser.add_argument("-k", "--keyword", default=None, help="Full‑text keyword on raw_text")
    parser.add_argument("-d", "--date-from", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("-D", "--date-to", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("-a", "--amount-min", type=float, default=None, help="Min amount")
    parser.add_argument("-A", "--amount-max", type=float, default=None, help="Max amount")
    parser.add_argument("-l", "--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    print(f"Query: {args.query}")
    if args.vendor:
        print(f"Vendor filter: {args.vendor}")
    if args.keyword:
        print(f"Keyword filter: {args.keyword}")
    print(f"Limit: {args.limit}\n")

    results = search_similar(
        args.query,
        vendor_filter=args.vendor,
        keyword_filter=args.keyword,
        date_from=args.date_from,
        date_to=args.date_to,
        amount_min=args.amount_min,
        amount_max=args.amount_max,
        top_k=args.limit
    )

    if not results:
        print("No matching invoices found.")
    else:
        print(f"{'ID':<5} {'Inv #':<20} {'Date':<12} {'Vendor':<25} {'Amount':<12} {'Sim':<8}")
        print("-" * 85)
        for inv_id, inv_num, date, vendor_name, amount, currency, sim in results:
            print(f"{inv_id:<5} {inv_num:<20} {date:<12} {vendor_name:<25} {amount:<12} {sim:.4f}")